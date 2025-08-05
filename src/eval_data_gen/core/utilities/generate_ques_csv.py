import os
import json
import csv
import random
import re

INPUT_DIR = os.path.join('tmp', 'questions')
OUTPUT_CSV = os.path.join('tmp', 'randomized_questions_with_answers.csv')

def parse_mcq_string(mcq_text):
    """
    Parses a single multi-line string containing a question, options, and answer.
    Returns a dictionary with structured data, or None if parsing fails.
    """
    try:
        pattern = re.compile(r"Q\d+\.\s*(.*?)\n((?:\(\d+\).*?\n)+)Answer:\s*(\d+)", re.DOTALL)
        match = pattern.search(mcq_text)

        if not match:
            return None

        question = match.group(1).strip()
        options_block = match.group(2).strip()
        answer_number = int(match.group(3)) # Store the original answer number
        answer_index = answer_number - 1    # Convert to 0-based index

        options = [opt.strip() for opt in options_block.split('\n')]
        cleaned_options = [re.sub(r'^\(\d+\)\s*', '', opt) for opt in options]

        if 0 <= answer_index < len(cleaned_options):
            answer_text = cleaned_options[answer_index]
        else:
            answer_text = "Error: Invalid Answer Index"

        return {
            "question": question,
            "options": cleaned_options,
            "answer_text": answer_text,
            "answer_number": answer_number
        }
    except Exception as e:
        print(f"Warning: Failed to parse a question string due to error: {e}\nString: '{mcq_text[:50]}...'")
        return None


def create_randomized_csv():
    """
    Reads all JSON files, parses the MCQs, randomizes them, 
    and writes to a single CSV file.
    """
    all_questions_structured = []

    if not os.path.isdir(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' not found.")
        return

    print(f"Reading JSON files from '{INPUT_DIR}'...")

    # Read JSON files and parse the MCQ strings
    for filename in os.listdir(INPUT_DIR):
        if filename.endswith('.json'):
            file_path = os.path.join(INPUT_DIR, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                    mcq_strings = data.get('mcqs', [])
                    for mcq_string in mcq_strings:
                        parsed_data = parse_mcq_string(mcq_string)
                        if parsed_data:
                            all_questions_structured.append(parsed_data)
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON from '{filename}'. Skipping.")

    if not all_questions_structured:
        print("No valid questions were found. Exiting.")
        return

    print(f"Found and parsed a total of {len(all_questions_structured)} questions.")

    # Randomize the entire list of questions
    random.shuffle(all_questions_structured)
    print("Successfully randomized all questions.")

    # Write the structured, randomized data to a CSV file
    try:
        with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as csvfile:
            max_options = max(len(q['options']) for q in all_questions_structured)
            header = ['Question Text'] + [f'Option {i+1}' for i in range(max_options)] + ['Answer Text', 'Answer Number']
            
            writer = csv.writer(csvfile)
            writer.writerow(header)

            for q_data in all_questions_structured:
                options_padded = q_data['options'] + [''] * (max_options - len(q_data['options']))
                # to add the new data to row
                row = [q_data['question']] + options_padded + [q_data['answer_text'], q_data['answer_number']]
                writer.writerow(row)
        
        print(f"\nSuccess! CSV file created at: {os.path.abspath(OUTPUT_CSV)}")

    except IOError as e:
        print(f"Error: Could not write to file '{OUTPUT_CSV}'. {e}")

# Run the main function
if __name__ == '__main__':
    create_randomized_csv()