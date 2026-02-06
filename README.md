## csvsurveybot

### Overview

Surveybot is an open-source **Streamlit-based app** for **LLM-assisted qualitative coding** of survey verbatims stored in a CSV. The app lets you generate themes from open-ended responses (reasoning step), classify responses against those themes, and export a CSV with 0/1 indicator columns per theme, all in a fairly self-explanatory ui. The app is a single .py file so should be fairly simple for even a non-technical user to download and run (perhaps with some LLM help).

Key features:
- The generate-then-classify workflow is designed to allow a researcher more control over theme coding than some other AI-led theme coding tools, allowing manual editing of themes as well as LLM-powered consolidation, merging, splitting and deduplication. 
- Additional instructions can be provided for all llm calls, e.g. to direct the llm to look for certain types of themes, or to create themes in a different language to the verbatim responses.
- Multiple theme sets can be defined based on the same column of data, enabling, for example, researchers to explore different 'additional instructions' for the same column.
- Each theme is stored with a longer theme description, and examples of the theme that have been validated as real verbatims, all of which are used for more accurate theme classification.
- Verbatim examples of theme, and a results table showing each row of data and the themes coded against it, provide an easy way to manually verify the quality of theme coding.
- A simple bar chart visualisation shows the prevalence of different themes at a glance.
- In addition to exporting a csv of coded data, you can export an 'all themes' summary table with theme descriptions and counts, import and export theme sets (e.g. to use a theme set on a new dataset), and save and load app state to resume earlier theme coding.
- The app runs entirely locally except for API calls to OpenAI (you provide the API key). Use of local LLM models is also supported via ollama.

---

### How to run

The entire app is a single Python file. To run it:

```bash
pip install streamlit openai langchain-ollama pandas pydantic streamlit-aggrid altair
streamlit run csvsurveybot2.py
```

This opens the app in your browser. Everything runs locally -- your data is not sent anywhere except to the LLM API you choose.

---

### LLM backends

The app supports two AI backends, selectable in the sidebar:

- **OpenAI** -- cloud models (GPT-5.2, o3, etc.). Requires an API key, which you enter in the sidebar. The app fetches your available models automatically.
- **Ollama** -- locally-hosted open models (currently set up to include qwen3:4b and gemma3:4b). Requires [Ollama](https://ollama.com) installed and running on your machine. No API key needed.

You can use different models for theme generation and classification (e.g. a more creative model for discovering themes, a cheaper/faster one for classification).

---

### Technical notes

- **Single-file architecture.** All UI, state management, LLM prompting, and export logic lives in `csvsurveybot2.py`.
- **Structured LLM output.** OpenAI calls use JSON schema mode with strict validation, backed by Pydantic models. Ollama calls fall back to JSON extraction with structural validation.
- **Separate model settings for reasoning vs. classification.** Historically, reasoning models like o3 have been more effective at reasoning (theme generation) stage, whilst more efficient models can be used without meaningful degradation of performance at classification stage.
- **Serial and parallel batch processing.** Theme generation can run serially (each batch builds on the previous themes) or in parallel (independent batches, then consolidated). Serial generation is preferable unless speed is a problem as it minimises the need to consolidate themes. Classification always runs in parallel with a configurable batch size.
- **Sparse result storage.** Classification results are stored as `{row_index: [theme_names]}` dictionaries rather than dense DataFrames, which keeps memory usage low for large datasets with many themes.
- **Token-aware context management.** The app tracks token usage per call against each model's context window and warns before you exceed limits.
- **Wide-format CSV export.** Output columns follow the pattern `{question}_{theme_set}_{theme}` with 0/1 values, designed to drop straight into cross-tabulation or statistical tools.
