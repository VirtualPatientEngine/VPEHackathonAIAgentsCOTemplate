# VPE Hackathon: AI Agents for Life Sciences
Please visit the project page [website](https://virtualpatientengine.github.io/VPEHackathonAIAgentsCO/) for a detailed overview of the Hackathon tasks and results

## ⚠️ Warning ⚠️
- The results of the Hackathon are provided as with only minimal changes by the judges made during the evaluation and which were made to reproduce the participants final demos given at the end of the Hackathon.
- The Hackathon code is currently specific to the Code Ocean capsule environment which was used to boost participant productivity and ensure reproducibility of results.
- We are in the process of migrating the Talk2Biomodels code to a stand-alone repository following our internal code- and dev-ops to facilitate community engagement (be on the lookout 👀 for when we announce the completion of the migration).

## Recommended environment setup options
1. From a code ocean capsule
2. From a Docker image locally or on the cloud
3. From a python virtual environment

## Running the projects
### Assumed folder structure
> code/... # where the code lives<br>
> data/... # where data assets live<br>
> environment/... # where docker files and other environment setup scripts live<br>
> results/... # where results are written to<br>

### From the command line
Run the demo [README](./code/demo/README.md)<br>
`streamlit run /code/demo/streamlit_app.py`<br>
Run the Talk2Biomodels [README](./code/talk2biomodels/README.md)<br>
`streamlit run /code/talk2biomodels/streamlit_app.py`<br>
Run the Talk2Cells [README](./code/talk2cells/README.md)<br>
`streamlit run /code/talk2cells/streamlit_app.py`<br>
Run the Talk2KnowledgeGraphs [README](./code/talk2knowledgegraphs/README.md)<br>
`streamlit run /code/talk2knowledgegraphs/streamlit_app.py`<br>

### Caveats
- Talk2Biomodels is currently not integrated with Ollama, and instead uses OpenAI which requires setting the environmental variable `OPENAI_API_KEY`
- Demo, Talk2Cells, and Talk2KnowledgeGraphs require external data assets that were supplied during the Hackathon, which are explained in the [setup](https://virtualpatientengine.github.io/VPEHackathonAIAgentsCO/setup/) and project specific READMEs