"""

 ****************************************************************************

    @file       create_sinhala_dataset.py

    @author     Achintha Ihalage

    @brief      This module is responsible for translating a subset of ELI5
                (explain like I'm five) dataset in English to Sinhala using
                Google Cloud Translation API.

 ****************************************************************************

"""

from google.cloud import translate_v2 as translate
from datasets import load_dataset, Dataset
import os
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

## setting up google cloud credentials
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'API_KEYS/YOUR_API_CREDENTIALS.json'

class SinhalaWriter(object):
    """
    A class to translate HF datasets in English to Sinhala.
    """
    def __init__(self,
                 source_language: str = "en",
                 target_language: str ="si",
                 dataset_name: str = "Pavithree/eli5",
                 limit_translation_len: bool = True,
                 max_n_chars_to_translate: int = 100000,
                 max_n_samples_to_translate: int = 100,
                 path_to_save_dataset: str = "eli5_sinhala.jsonl",
                 ) -> None:
        self.source_language = source_language
        self.target_language = target_language
        self.dataset_name = dataset_name
        self.limit_translation_len = limit_translation_len
        self.max_n_chars_to_translate = max_n_chars_to_translate
        self.max_n_samples_to_translate = max_n_samples_to_translate
        self.path_to_save_dataset = path_to_save_dataset
        self.translate_client = translate.Client()
        self.total_n_chars_translated = 0

    def load_HF_dataset(self,
                        ) -> Dataset:
        """
        A function to load a Hugging Face dataset.

        Returns:
            A loaded HF dataset. 
        """
        dataset = load_dataset(self.dataset_name)
        return dataset
    
    def count_total_n_chars_in_ELI5_dataset(self,
                                       dataset: Dataset) -> int:
        """
        Counts the total number of characters in the dataset (input text data).
        Helps to estimate the cost of translation.
        This function is specific to ELI5 dataset.

        Args:
            dataset (Dataset): A loaded HF dataset.
        
        Returns:
            Total character count.
        """

        total_n_chars = 0
        for i,entry in enumerate(dataset['train']):
            if i%1000 == 0:
                print(f"Processing entry No: {i}")
            question = entry['title']
            answer = entry['answers']['text'][0] if len(entry['answers']['text'])>0 else "This question is niche or too difficult to answer!"
            total_n_chars += (len(question) + len(answer))

        print(f"Number of samples in the dataset: {len(dataset)}")
        print(f"Total number of characters in the dataset: {total_n_chars}")

        return total_n_chars
    
    def translate_text(self, 
                       text: str) -> str:
        """
        Translate text from source language to target language using Google Cloud Translate API.

        Args:
            text (str): The text to be translated.

        Returns:
            str: The translated text.
        """

        ## we want to limit the translation to some max number of input chars to avoid large costs for translate API.
        if self.limit_translation_len and self.total_n_chars_translated + len(text) > self.max_n_chars_to_translate:
            raise StopIteration("Reached the maximum number of characters to translate.")
        
        result = self.translate_client.translate(text, target_language=self.target_language)
        self.total_n_chars_translated += len(text)
        return result['translatedText']
    
    def translate_dataset_ELI5(self, 
                          dataset: Dataset) -> Dataset:
        """
        Translate all questions and answers in the dataset to the target language.
        This function is specific to ELI5 (explain like I'm 5) dataset due to its distinct format.

        Args:
            dataset (Dataset): The dataset to be translated in HF Datasets format.

        Returns:
            Dataset: The translated dataset.
        """
        translated_data = []

        for i, entry in enumerate(dataset['train']):
            if i % 1000 == 0 and i>0:
                logger.info(f"Translating entry No: {i}")
                logger.info(f"Writing dataset checkpoint ==> {self.path_to_save_dataset.replace('.jsonl', f'_{len(dataset)}.jsonl')}\n")
                self.save_translated_dataset(Dataset.from_list(translated_data))
            
            try:
                if self.limit_translation_len and i==self.max_n_samples_to_translate:
                    raise StopIteration("Reached the maximum number of samples to translate.")
                
                question = entry['title']
                # body = entry['selftext'] ## this is the body of a question, which could be important
                answer = entry['answers']['text'][0] if len(entry['answers']['text'])>0 else "This question is niche or too difficult to answer!"
                translated_question = self.translate_text(question)
                translated_answer = self.translate_text(answer)
                translated_data.append({
                    'q_id': entry['q_id'],
                    'subreddit': entry['subreddit'],
                    'url': entry['url'],
                    'sinhala_question': translated_question,
                    'sinhala_answer': translated_answer,
                    'english_question': question,
                    'english_answer': answer
                })
            except StopIteration:
                logger.info(f"Either maximum translation character count ({self.max_n_chars_to_translate}) "
                            f"or maximum number of samples ({self.max_n_samples_to_translate}) is reached at entry No: {i}")
                break
        # print(translated_data)
        # print(translated_data[0])
        return Dataset.from_list(translated_data)
    
    def save_translated_dataset(self, 
                                dataset: Dataset, 
                                ) -> None:
        """
        Save the translated dataset to a JSON file.

        Args:
            dataset (Dataset): The dataset to be saved.
            output_path (str): The path to save the translated dataset.
        """

        ## modify filename to have the number of translated samples
        modified_path = self.path_to_save_dataset.replace(".jsonl", f"_{len(dataset)}.jsonl")
        with open(modified_path, 'w', encoding='utf-8') as f:
            for entry in dataset:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    def execute_translation_job(self,
                                ) -> None:
        """
        Executes the entire translation job. Saves the resulting dataset to the disk.
        """

        dataset = self.load_HF_dataset()
        # total_chars = self.count_total_n_chars_in_ELI5_dataset(dataset)
        translated_dataset = self.translate_dataset_ELI5(dataset)
        logging.info(f"A total number of {self.total_n_chars_translated} characters have been translated!")
        
        
        self.save_translated_dataset(translated_dataset)

    ## utility function. redundant
    def rearrange_HF_dataset(self,
                             dataset_path: str) -> None:
        """
        Adds English question and English answer to the dataset
        """
        dataset = load_dataset('json', data_files=dataset_path)
        def rename_columns(example):
            return {
                "sinhala_question": example["question"],
                "sinhala_answer": example["answer"]
            }

        # apply the renaming function to the dataset
        dataset = dataset.map(rename_columns, remove_columns=["question", "answer"])
        
        eli5_dataset = self.load_HF_dataset()
        eli5_subset = eli5_dataset['train'].select(range(len(dataset['train'])))
        print(eli5_subset)
        english_questions = []
        english_answers = []

        # Process each entry to apply the condition
        for entry in eli5_subset:
            english_questions.append(entry['title'])
            if len(entry['answers']['text']) > 0:
                english_answers.append(entry['answers']['text'][0])
            else:
                english_answers.append("This question is niche or too difficult to answer!")

        print(len(english_questions), len(english_answers))
        print(dataset['train'])
        dataset['train'] = dataset['train'].add_column('english_question', english_questions)
        dataset['train'] = dataset['train'].add_column('english_answer', english_answers)

        print(dataset)
        print(dataset['train'])
        with open('translated_datasets/sinhala-finetune-qa-eli5.jsonl', 'w', encoding='utf-8') as f:
            for entry in dataset['train']:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')



if __name__=="__main__":
    output_path = 'translated_datasets/eli5_sinhala.jsonl'
    sinhala_writer = SinhalaWriter(limit_translation_len=True,
                                   max_n_samples_to_translate=100,
                                   max_n_chars_to_translate=160000,
                                   path_to_save_dataset=output_path)
    sinhala_writer.execute_translation_job()

        