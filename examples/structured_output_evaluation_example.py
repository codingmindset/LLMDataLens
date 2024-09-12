import os
from llmdatalens import (
    StructuredOutputEvaluator,
    LLMStructuredOutput,
    GroundTruth,
    Metadata,
    Prompt,
    FunctionSchema,
    ExperimentManager,
    MetricNames
)

# Ensure the OpenAI API key is set
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable")

# Sample data
golden_data = {
    'number': 'INV-2023-12345',  # Invoice number
    'date': 'Sept 5, 2024', 
    'incoterm': 'CFR', 
    'currency': 'EUR',
    'customer_name': 'Acme Corporation', 
    'items': [
        {'name': 'Laptop - Model XYZ', 'price': 899.99, 'code': 'LP-XYZ', 'quantity': 2},
        {'name': 'Wireless Mouse', 'price': 24.95, 'code': 'WM-001', 'quantity': 5},
        {'name': 'External Hard Drive - 2TB', 'price': 79.50, 'code': 'HD-2TB', 'quantity': 1},
    ],
    'total': 2004.43  # Updated total based on new items
}


# Simulated LLM output (in this case, it's identical to the golden data for simplicity)
llm_result = golden_data.copy()

# llm result with a different total
llm_result['total'] = 2004.43
llm_result['customer_name'] = 'Acme'    

# Create ground truth
ground_truth = GroundTruth(data=golden_data)

# Create an evaluator with specific metrics
evaluator = StructuredOutputEvaluator(
    metrics=[MetricNames.OverallAccuracy, MetricNames.AverageLatency],
    experiment_name="Invoice Processing Experiment",
    experiment_version="1.0.0",
    openai_api_key=openai_api_key
)

# Define function schema
function_schema = FunctionSchema(
    name="Invoice",
    description="An invoice containing items purchased by a customer and the total",
    parameters={
        'type': 'object',
        'properties': {
            'number': {'description': 'the number of the invoice', 'type': 'string'},
            'date': {'description': 'the date of the invoice', 'type': 'string'},
            'incoterm': {
                'description': 'the delivery condition of the invoice',
                'allOf': [{
                    'title': 'Incoterm',
                    'description': 'An enumeration.',
                    'enum': [
                        'FAS – Free Alongside Ship (named port of shipment)',
                        'FOB – Free on Board (named port of shipment)',
                        'CFR – Cost and Freight (named port of destination)',
                        'CIF – Cost and Invoice (named port of destination)'
                    ],
                    'type': 'string'
                }]
            },
            'currency': {
                'description': 'the currency of the invoice',
                'allOf': [{
                    'title': 'Currency',
                    'description': 'An enumeration.',
                    'enum': ['USD', 'EUR', 'GBP'],
                    'type': 'string'
                }]
            },
            'customer_name': {'type': 'string'},
            'items': {
                'type': 'array',
                'items': {
                    'description': 'A list of items purchased',
                    'type': 'object',
                    'properties': {
                        'name': {'description': 'the name or description of the item', 'type': 'string'},
                        'price': {'description': 'the individual price of the item (without currency symbol)', 'type': 'number'},
                        'code': {'description': 'the number or reference of the item', 'type': 'string'},
                        'quantity': {'description': 'the quantity of the item', 'type': 'integer'}
                    },
                    'required': ['name', 'price', 'code', 'quantity']
                }
            },
            'total': {'description': 'the total amount of the invoice', 'type': 'number'}
        },
        'required': ['number', 'date', 'incoterm', 'currency', 'customer_name', 'items', 'total']
    }
)

# Create LLM output
llm_output = LLMStructuredOutput(
    output_type="structured",
    structured_output=llm_result,
    metadata=Metadata(
        model_name="gpt-4o-mini",
        model_version="1.0",
        prompt=Prompt(
            system="You are an AI assistant specialized in extracting information from invoices.",
            # user="Please extract the details from the following invoice:",
            function_call=function_schema
        ),
        latency=0.5,
        confidence=0.9
    )
)

# Add LLM output and ground truth to the evaluator
evaluator.add_llm_output(llm_output)
evaluator.add_ground_truth(ground_truth)

# Evaluate
evaluation_result = evaluator.evaluate()

# Print results
print("Evaluation Result:")
print(f"Overall Field Accuracy: {evaluation_result.overall_accuracy:.2f}")
print("\nField Results:")
for field_name, field_result in evaluation_result.field_results.items():
    print(f"  {field_name}: {'Correct' if field_result.correct else 'Incorrect'}")
    print(f"    Predicted: {field_result.predicted}")
    print(f"    Ground Truth: {field_result.ground_truth}")
    if field_result.details:
        if "array_accuracy" in field_result.details:
            print(f"    Array Accuracy: {field_result.details['array_accuracy']:.2f}")
            print(f"    Correct Items: {field_result.details['correct_items']}")
            print(f"    Total Items: {field_result.details['total_items']}")
            print("    Item Results:")
            for i, item_result in enumerate(field_result.details['item_results']):
                print(f"      Item {i+1}: {'Correct' if item_result.correct else 'Incorrect'}")
        elif "relevancy_score" in field_result.details:
            print(f"    Relevancy Score: {field_result.details['relevancy_score']:.2f}")
            print(f"    Reason: {field_result.details['reason']}")
            print("    Statements:")
            for statement in field_result.details['statements']:
                print(f"      - {statement}")
            print("    Relevant Statements:")
            for statement in field_result.details['relevant_statements']:
                print(f"      - {statement}")
        else:
            print(f"    Details: {field_result.details}")

# Access experiment data
experiment_manager = ExperimentManager()
experiment = experiment_manager.get_experiment(evaluator.experiment_id)
print(f"\nExperiment: {experiment.name}")
print(f"Version: {experiment.version}")
print(f"Number of runs: {len(experiment.runs)}")

# Print details of the first run
if experiment.runs:
    first_run = experiment.runs[0]
    print("\nFirst Run Details:")
    print(f"Run ID: {first_run.id}")
    print(f"Model: {first_run.llm_output.metadata.model_name} (version: {first_run.llm_output.metadata.model_version})")
    print(f"Evaluation Metrics:")
    print(f"  Overall Accuracy: {first_run.evaluation_result.overall_accuracy:.2f}")
    for field_name, field_result in first_run.evaluation_result.field_results.items():
        print(f"  {field_name}: {'Correct' if field_result.correct else 'Incorrect'}")
