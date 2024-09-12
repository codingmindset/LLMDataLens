import json
from pydantic import BaseModel, Field
from openai import OpenAI
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class LLMEvaluator(BaseModel):
    model: str = Field(default="gpt-4o-mini")
    api_key: str = Field(default=None)

    def evaluate_relevancy(self, input_text: str, actual_output: str) -> Dict[str, Any]:
        client = OpenAI(api_key=self.api_key)
        prompt = f"""
        Input: {input_text}
        Actual Output: {actual_output}

        Task:
         1. Extract all statements made in the Actual Output.
         2. For each statement, determine if it is relevant to the Input.
         3. Calculate the relevancy score as: (Number of Relevant Statements) / (Total Number of Statements)
         4. Provide a brief reason for the score.

        Format your response as a JSON object with the following keys:
        - statements: A list of all extracted statements
        - relevant_statements: A list of statements deemed relevant
        - relevancy_score: The calculated relevancy score
        - reason: A brief explanation for the score
        """

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an AI assistant tasked with evaluating the relevancy of an output to a given input."},
                    {"role": "user", "content": prompt}
                ]
            )

            content = response.choices[0].message.content

            logger.debug(f"API Response: {response}")
            logger.debug(f"Content type: {type(content)}")
            logger.debug(f"Content: {content}")

            if not content:
                raise ValueError("Empty response from LLM")

            # Remove any markdown formatting
            content = content.strip('`').strip()
            if content.startswith('json'):
                content = content[4:].strip()

            # Parse the JSON content
            result = json.loads(content)
            return result

        except json.JSONDecodeError as json_error:
            error_message = f"Invalid JSON response: {content}. Error: {str(json_error)}"
        except Exception as e:
            error_message = f"LLM evaluation failed: {str(e)}"

        return {
            "error": error_message,
            "statements": [],
            "relevant_statements": [],
            "relevancy_score": 0,
            "reason": "Evaluation failed due to an error"
        }