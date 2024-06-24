## Setting Up

Please follow the instructions from [Google Cloud](https://cloud.google.com/python/docs/setup#linux) for installing the google cloud python API. Log In to your google cloud account and download your credentials as a JSON file and store it locally. Add your credentials to `os.environ['GOOGLE_APPLICATION_CREDENTIALS']` (line 27).

Install the `datasets` package from Hugging Face.

`pip install datasets`

## Dataset Card for sinhala-eli5
Sinhala question answering (QA) dataset contains a subset of the translated eli5 (explain like I'm 5) English dataset. eli5 is a crowdsourced dataset based mainly on the content from the subreddit r/explainlikeimfive. This is a forum where users post complex questions and other users provide simplified explanations.

A subset of eli5 dataset (5k/10k samples) has been machine translated to Sinhala language using the Google Cloud Translation API. The dataset is also [hosted](https://huggingface.co/datasets/ihalage/sinhala-finetune-qa-eli5) on Hugging Face.

## Dataset Format
sinhala-eli5 dataset contains the following columns.

- q_id (question ID)
- subreddit (subreddit where the question was asked)
- url (URL to the question)
- sinhala_question (Sinhala translation of the question)
- sinhala_answer (Sinhala translation of an answer to the question)
- english_question (Original English question)
- english_answer (an English answer)

## Applications
Many open source LLMs do not perform well on relatively uncommon languages such as Sinhala (as of June 2024). This dataset can be used to finetune open-source LLMs for Sinhala language, allowing the models to learn the syntax and semantics of the language and be able to generate coherent and contextually appropriate text in Sinhala. As the dataset contains the original English text too, one may use this dataset to finetune LLMs for translation tasks.

- Sinhala question answering
- English-Sinhala translation
