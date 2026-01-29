"""LLM-based commentary generation (optional feature)."""

import logging
import os
from typing import Dict, Optional, List
from pathlib import Path
import json

logger = logging.getLogger(__name__)


def generate_commentary(
    context: Dict,
    config: Dict,
    output_path: Optional[Path] = None
) -> str:
    """
    Generate automated market commentary using LLM.
    
    Args:
        context: Dictionary with forecast data, metrics, drivers, etc.
        config: LLM configuration from reporting.yaml
        output_path: Optional path to save commentary
    
    Returns:
        Generated commentary text
    """
    llm_config = config["reporting"]["llm_settings"]
    
    if not llm_config.get("enabled", False):
        logger.info("LLM commentary disabled in config")
        return "LLM commentary disabled"
    
    # Load API key from environment
    provider = llm_config["provider"]
    api_key = _get_api_key(provider)
    
    if not api_key:
        logger.warning(f"No API key found for {provider}, skipping commentary")
        return "No API key configured"
    
    # Load prompt template
    template_path = Path(llm_config["prompt_template"])
    prompt = _build_prompt(template_path, context, llm_config)
    
    # Call LLM
    logger.info(f"Calling {provider} API for commentary generation")
    
    try:
        if provider == "openai":
            commentary = _call_openai(prompt, llm_config, api_key)
        elif provider == "anthropic":
            commentary = _call_anthropic(prompt, llm_config, api_key)
        elif provider == "gemini":
            commentary = _call_gemini(prompt, llm_config, api_key)
        else:
            raise ValueError(f"Unknown LLM provider: {provider}")
        
        # Log interaction
        _log_llm_interaction(prompt, commentary, config, output_path)
        
        # Save commentary
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(commentary)
            logger.info(f"Commentary saved to {output_path}")
        
        return commentary
    
    except Exception as e:
        logger.error(f"Failed to generate commentary: {e}")
        return f"Error generating commentary: {e}"


def _get_api_key(provider: str) -> Optional[str]:
    """Get API key from environment."""
    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }
    
    env_var = key_map.get(provider)
    return os.getenv(env_var)


def _build_prompt(
    template_path: Path,
    context: Dict,
    llm_config: Dict
) -> str:
    """Build prompt from template and context."""
    if not template_path.exists():
        logger.warning(f"Template not found: {template_path}")
        return _default_prompt(context)
    
    # Load template
    with open(template_path, 'r') as f:
        template = f.read()
    
    # Format with context
    prompt = template.format(**context)
    
    return prompt


def _default_prompt(context: Dict) -> str:
    """Generate default prompt if template not found."""
    return f"""
Generate a brief market commentary for power prices based on the following information:

Forecast Summary:
{context.get('forecast_summary', 'N/A')}

Model Performance:
{context.get('model_performance', 'N/A')}

Key Drivers:
{context.get('key_drivers', 'N/A')}

Market Conditions:
{context.get('market_conditions', 'N/A')}

Please provide:
1. Brief overview of price forecast
2. Key factors driving prices
3. Any notable risks or uncertainties
4. Trading implications

Keep the commentary concise (2-3 paragraphs).
"""


def _call_openai(prompt: str, config: Dict, api_key: str) -> str:
    """Call OpenAI API."""
    try:
        import openai
        
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model=config["model"],
            messages=[
                {"role": "system", "content": "You are a power market analyst providing concise, professional commentary on electricity price forecasts."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
        )
        
        return response.choices[0].message.content
    
    except ImportError:
        logger.error("openai package not installed. Install with: pip install openai")
        return "OpenAI package not available"
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        return f"API error: {e}"


def _call_anthropic(prompt: str, config: Dict, api_key: str) -> str:
    """Call Anthropic API."""
    try:
        import anthropic
        
        client = anthropic.Anthropic(api_key=api_key)
        
        response = client.messages.create(
            model=config["model"],
            max_tokens=config["max_tokens"],
            temperature=config["temperature"],
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.content[0].text
    
    except ImportError:
        logger.error("anthropic package not installed. Install with: pip install anthropic")
        return "Anthropic package not available"
    except Exception as e:
        logger.error(f"Anthropic API error: {e}")
        return f"API error: {e}"


def _call_gemini(prompt: str, config: Dict, api_key: str) -> str:
    """Call Google Gemini API using google-generativeai."""
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name=config["model"])
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=config["max_tokens"],
                temperature=config["temperature"],
            )
        )
        return response.text
    except ImportError:
        logger.error("google-generativeai package not installed. Install with: pip install google-generativeai")
        return "Gemini package not available"
    except Exception as e:
        logger.error(f"Gemini API error: {e}")
        return f"API error: {e}"


def _log_llm_interaction(
    prompt: str,
    response: str,
    config: Dict,
    output_path: Optional[Path]
):
    """Log LLM interaction for transparency."""
    from datetime import datetime
    
    log_dir = Path(config["reporting"]["output_paths"]["report_dir"]) / "llm_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / f"llm_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "provider": config["reporting"]["llm_settings"]["provider"],
        "model": config["reporting"]["llm_settings"]["model"],
        "prompt": prompt,
        "response": response,
        "output_path": str(output_path) if output_path else None,
    }
    
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)
    
    logger.info(f"LLM interaction logged to {log_file}")
