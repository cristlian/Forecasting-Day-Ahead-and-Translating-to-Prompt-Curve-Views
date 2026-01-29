"""
Trading Agent: Automated Morning Trading Signal Generator
=========================================================

This module implements an AI-powered trading agent that acts as a Senior Trader,
generating actionable execution strategies from forecasts and Clean Spark Spreads.

Key Features:
- Senior Trader persona for professional, actionable outputs
- 3-bullet execution strategy format (desk-ready)
- Risk-aware signal generation with invalidation rules
- Integration with LLM providers (Gemini, OpenAI, Anthropic)

This is Step 9 of the pipeline - "Agent-based workflow development"
"""

import logging
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# TRADING AGENT CONFIGURATION
# ============================================================================

AGENT_CONFIG = {
    "persona": "Senior Power Trader",
    "desk": "European Power Trading Desk",
    "market": "DE-LU Day-Ahead",
    "style": "Direct, actionable, risk-aware",
}

SENIOR_TRADER_SYSTEM_PROMPT = """You are a Senior Power Trader on the European Power Trading Desk with 15+ years of experience in German power markets.

Your role:
- Generate ACTIONABLE execution strategies for the morning desk
- Be DIRECT and SPECIFIC - traders need concrete actions, not vague commentary
- Always include position sizing guidance (scale: small/medium/full)
- Flag risks clearly but don't be paralyzed by them
- Think in terms of EUR P&L impact

Your output format MUST be exactly 3 bullets:
1. **POSITION**: What to do (BUY/SELL/HOLD), which bucket, and size
2. **RATIONALE**: Why (1-2 sentences max, reference CSS/drivers)
3. **RISK**: Key risk to monitor and action if triggered

Keep each bullet to 1-2 lines maximum. No fluff. Traders are busy."""


# ============================================================================
# MAIN AGENT FUNCTION
# ============================================================================

def generate_morning_signal(
    signals_df: pd.DataFrame,
    bucket_df: pd.DataFrame,
    config: Dict[str, Any],
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Generate automated morning trading signal using LLM as Senior Trader.
    
    Args:
        signals_df: Hourly signals with CSS calculations
        bucket_df: Bucket-level aggregates (Peak/Off-Peak/Weekend)
        config: Pipeline configuration
        output_dir: Directory to save outputs
        
    Returns:
        Dictionary containing:
        - strategy: The 3-bullet execution strategy
        - context: Data context used
        - metadata: Generation metadata
    """
    logger.info("🤖 Trading Agent: Generating morning signal...")
    
    # Build context from data
    context = _build_trading_context(signals_df, bucket_df)
    
    # Build the prompt
    prompt = _build_agent_prompt(context)
    
    # Get LLM settings
    llm_config = config.get("reporting", {}).get("llm_settings", {})
    
    # Get market code from config
    market = config.get("market", {}).get("market", {}).get("code", "DE-LU")
    
    if not llm_config.get("enabled", True):
        logger.warning("LLM disabled in config, using fallback strategy")
        strategy = _fallback_strategy(context)
    else:
        strategy = _call_llm_agent(prompt, llm_config)
    
    # Package result
    result = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "market": market,
        "period": f"{signals_df.index.min()} to {signals_df.index.max()}",
        "strategy": strategy,
        "context": context,
        "metadata": {
            "agent_persona": AGENT_CONFIG["persona"],
            "llm_provider": llm_config.get("provider", "fallback"),
            "llm_model": llm_config.get("model", "N/A"),
        }
    }
    
    # Save outputs
    if output_dir:
        _save_agent_outputs(result, output_dir)
    
    return result


# ============================================================================
# CONTEXT BUILDING
# ============================================================================

def _build_trading_context(
    signals_df: pd.DataFrame,
    bucket_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Build structured context for the trading agent."""
    
    # Overall statistics
    avg_css = signals_df["clean_spark_spread"].mean()
    profitable_hours = (signals_df["clean_spark_spread"] > 0).sum()
    total_hours = len(signals_df)
    profitable_pct = profitable_hours / total_hours * 100
    
    # Dispatch P&L
    total_dispatch_pnl = signals_df.get("dispatch_pnl", signals_df["clean_spark_spread"].clip(lower=0)).sum()
    
    # Price statistics
    avg_price = signals_df["predicted_price"].mean()
    max_price = signals_df["predicted_price"].max()
    min_price = signals_df["predicted_price"].min()
    
    # Signal distribution
    signal_counts = signals_df["signal"].value_counts().to_dict()
    
    # Bucket analysis
    bucket_summary = []
    for _, row in bucket_df.iterrows():
        bucket_summary.append({
            "bucket": row["bucket"],
            "price": row["predicted_price"],
            "css": row["clean_spark_spread"],
            "signal": row["margin_signal"],
        })
    
    # Find best and worst buckets
    best_bucket = bucket_df.loc[bucket_df["clean_spark_spread"].idxmax()]
    worst_bucket = bucket_df.loc[bucket_df["clean_spark_spread"].idxmin()]
    
    # Risk indicators
    negative_css_streak = _count_max_negative_streak(signals_df["clean_spark_spread"])
    price_volatility = signals_df["predicted_price"].std()
    
    context = {
        "forecast_date": signals_df.index.max().strftime("%Y-%m-%d"),
        "hours_analyzed": total_hours,
        "avg_css_eur": round(avg_css, 2),
        "profitable_hours": profitable_hours,
        "profitable_pct": round(profitable_pct, 1),
        "total_dispatch_pnl_eur": round(total_dispatch_pnl, 2),
        "avg_price_eur": round(avg_price, 2),
        "price_range": f"€{min_price:.0f} - €{max_price:.0f}",
        "price_volatility": round(price_volatility, 2),
        "signal_distribution": signal_counts,
        "bucket_summary": bucket_summary,
        "best_bucket": {
            "name": best_bucket["bucket"],
            "css": round(best_bucket["clean_spark_spread"], 2),
        },
        "worst_bucket": {
            "name": worst_bucket["bucket"],
            "css": round(worst_bucket["clean_spark_spread"], 2),
        },
        "risk_indicators": {
            "max_negative_streak": negative_css_streak,
            "volatility_eur": round(price_volatility, 2),
            "low_confidence": profitable_pct < 60,
        },
        "marginal_cost": round(signals_df.get("marginal_cost", pd.Series([90])).iloc[0], 2),
    }
    
    return context


def _count_max_negative_streak(series: pd.Series) -> int:
    """Count maximum consecutive negative values."""
    negative = series < 0
    streak = 0
    max_streak = 0
    for val in negative:
        if val:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


# ============================================================================
# PROMPT BUILDING
# ============================================================================

def _build_agent_prompt(context: Dict[str, Any]) -> str:
    """Build the prompt for the Senior Trader agent."""
    
    # Format bucket summary
    bucket_lines = []
    for b in context["bucket_summary"]:
        bucket_lines.append(f"  - {b['bucket']}: Price €{b['price']:.0f}/MWh, CSS €{b['css']:.0f}/MWh → {b['signal']}")
    bucket_text = "\n".join(bucket_lines)
    
    prompt = f"""## MORNING BRIEFING: {context['forecast_date']}

### FORECAST SUMMARY
- Period: Next {context['hours_analyzed']} hours
- Average Price: €{context['avg_price_eur']}/MWh (Range: {context['price_range']})
- Price Volatility: €{context['price_volatility']}/MWh

### CLEAN SPARK SPREAD (CSS) ANALYSIS
- Marginal Cost (CCGT): €{context['marginal_cost']}/MWh
- Average CSS: €{context['avg_css_eur']}/MWh
- Profitable Hours: {context['profitable_hours']}/{context['hours_analyzed']} ({context['profitable_pct']}%)
- Total Dispatch P&L (1MW): €{context['total_dispatch_pnl_eur']}

### BUCKET VIEW
{bucket_text}

**Best Opportunity:** {context['best_bucket']['name']} (CSS: €{context['best_bucket']['css']}/MWh)
**Highest Risk:** {context['worst_bucket']['name']} (CSS: €{context['worst_bucket']['css']}/MWh)

### SIGNAL DISTRIBUTION
- BUY signals: {context['signal_distribution'].get('BUY', 0)} hours
- SELL signals: {context['signal_distribution'].get('SELL', 0)} hours
- HOLD signals: {context['signal_distribution'].get('HOLD', 0)} hours

### RISK FLAGS
- Max negative CSS streak: {context['risk_indicators']['max_negative_streak']} consecutive hours
- Price volatility: €{context['risk_indicators']['volatility_eur']}/MWh
- Low confidence mode: {'YES ⚠️' if context['risk_indicators']['low_confidence'] else 'NO ✓'}

---

**YOUR TASK:** Given this forecast and spark spread analysis, write a 3-bullet execution strategy for the morning desk.
"""
    
    return prompt


# ============================================================================
# LLM CALLS
# ============================================================================

def _call_llm_agent(prompt: str, llm_config: Dict) -> str:
    """Call LLM with Senior Trader persona."""
    
    provider = llm_config.get("provider", "gemini")
    api_key = _get_api_key(provider)
    
    if not api_key:
        logger.warning(f"No API key for {provider}, using fallback")
        return _fallback_strategy_text()
    
    logger.info(f"Calling {provider} API with Senior Trader persona...")
    
    try:
        if provider == "gemini":
            return _call_gemini_agent(prompt, llm_config, api_key)
        elif provider == "openai":
            return _call_openai_agent(prompt, llm_config, api_key)
        elif provider == "anthropic":
            return _call_anthropic_agent(prompt, llm_config, api_key)
        else:
            logger.error(f"Unknown provider: {provider}")
            return _fallback_strategy_text()
    except Exception as e:
        logger.error(f"LLM call failed: {e}")
        return _fallback_strategy_text()


def _get_api_key(provider: str) -> Optional[str]:
    """Get API key from environment."""
    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    return os.getenv(key_map.get(provider, ""))


def _call_gemini_agent(prompt: str, config: Dict, api_key: str) -> str:
    """Call Gemini API with Senior Trader persona."""
    try:
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=api_key)
        
        full_prompt = SENIOR_TRADER_SYSTEM_PROMPT + "\n\n" + prompt
        
        response = client.models.generate_content(
            model=config.get("model", "gemini-2.0-flash"),
            contents=full_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=config.get("max_tokens", 500),
                temperature=config.get("temperature", 0.3),
            )
        )
        
        return response.text
        
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        raise


def _call_openai_agent(prompt: str, config: Dict, api_key: str) -> str:
    """Call OpenAI API with Senior Trader persona."""
    try:
        import openai
        
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=config.get("model", "gpt-4"),
            messages=[
                {"role": "system", "content": SENIOR_TRADER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=config.get("max_tokens", 500),
            temperature=config.get("temperature", 0.3),
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise


def _call_anthropic_agent(prompt: str, config: Dict, api_key: str) -> str:
    """Call Anthropic API with Senior Trader persona."""
    try:
        import anthropic
        
        client = anthropic.Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model=config.get("model", "claude-3-sonnet-20240229"),
            max_tokens=config.get("max_tokens", 500),
            temperature=config.get("temperature", 0.3),
            system=SENIOR_TRADER_SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text
        
    except Exception as e:
        logger.error(f"Anthropic API error: {e}")
        raise


# ============================================================================
# FALLBACK STRATEGY (NO LLM)
# ============================================================================

def _fallback_strategy(context: Dict) -> str:
    """Generate rule-based strategy when LLM unavailable."""
    
    avg_css = context["avg_css_eur"]
    profitable_pct = context["profitable_pct"]
    best_bucket = context["best_bucket"]
    worst_bucket = context["worst_bucket"]
    low_confidence = context["risk_indicators"]["low_confidence"]
    
    # Position logic
    if avg_css > 20:
        position = f"**POSITION**: BUY {best_bucket['name']} block, FULL SIZE. Average CSS €{avg_css}/MWh supports baseload dispatch."
    elif avg_css > 0:
        position = f"**POSITION**: BUY {best_bucket['name']} block only, MEDIUM SIZE. Marginal economics - be selective."
    else:
        position = f"**POSITION**: HOLD/REDUCE exposure. Negative average CSS (€{avg_css}/MWh) - generation uneconomic."
    
    # Rationale
    rationale = f"**RATIONALE**: {profitable_pct}% of hours profitable. Best opportunity in {best_bucket['name']} (CSS €{best_bucket['css']}/MWh). Avoid {worst_bucket['name']} (CSS €{worst_bucket['css']}/MWh)."
    
    # Risk
    if low_confidence:
        risk = f"**RISK**: LOW CONFIDENCE MODE. Only {profitable_pct}% profitable hours. Cut position 50% if spot deviates >€15/MWh from forecast."
    else:
        risk = f"**RISK**: Monitor wind/solar actuals vs forecast. Invalidate signal if deviation >2GW intraday."
    
    return f"1. {position}\n2. {rationale}\n3. {risk}"


def _fallback_strategy_text() -> str:
    """Static fallback when even context is unavailable."""
    return """1. **POSITION**: HOLD - await LLM analysis. Reduce to half size until strategy confirmed.
2. **RATIONALE**: Automated agent unavailable. Manual review required before taking positions.
3. **RISK**: No automated risk assessment. Apply standard desk limits."""


# ============================================================================
# OUTPUT SAVING
# ============================================================================

def _save_agent_outputs(result: Dict, output_dir: Path):
    """Save agent outputs to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON with full context
    json_path = output_dir / f"morning_signal_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Signal JSON saved to {json_path}")
    
    # Save markdown report
    md_path = output_dir / f"morning_signal_{timestamp}.md"
    md_content = _format_signal_markdown(result)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Signal report saved to {md_path}")
    
    # Save latest (overwrite) for easy access
    latest_path = output_dir / "LATEST_MORNING_SIGNAL.md"
    with open(latest_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    logger.info(f"Latest signal saved to {latest_path}")


def _format_signal_markdown(result: Dict) -> str:
    """Format result as markdown report."""
    ctx = result["context"]
    
    return f"""# 🚨 MORNING TRADING SIGNAL
## {result['market']} | {ctx['forecast_date']}

**Generated:** {result['timestamp']}  
**Agent:** {result['metadata']['agent_persona']}  
**Model:** {result['metadata']['llm_model']}

---

## EXECUTION STRATEGY

{result['strategy']}

---

## SUPPORTING DATA

| Metric | Value |
|--------|-------|
| Forecast Period | {ctx['hours_analyzed']} hours |
| Average Price | €{ctx['avg_price_eur']}/MWh |
| Price Range | {ctx['price_range']} |
| Average CSS | €{ctx['avg_css_eur']}/MWh |
| Profitable Hours | {ctx['profitable_hours']} ({ctx['profitable_pct']}%) |
| Dispatch P&L (1MW) | €{ctx['total_dispatch_pnl_eur']} |

### Bucket Analysis

| Bucket | Predicted Price | CSS | Signal |
|--------|-----------------|-----|--------|
"""  + "\n".join([
        f"| {b['bucket']} | €{b['price']:.0f}/MWh | €{b['css']:.0f}/MWh | {b['signal']} |"
        for b in ctx['bucket_summary']
    ]) + f"""

### Risk Indicators

- Max negative CSS streak: {ctx['risk_indicators']['max_negative_streak']} hours
- Price volatility: €{ctx['risk_indicators']['volatility_eur']}/MWh
- Low confidence mode: {'⚠️ YES' if ctx['risk_indicators']['low_confidence'] else '✓ NO'}

---

*This signal is generated by an automated Trading Agent. Always apply human judgment before execution.*
"""
