"""Placeholder for structured_greedy_shuffled experiment."""
from shared import prompts, csv_writer

def run():
    prompt = prompts.get_prompt('structured','greedy_shuffled')
    print('Running', prompt)
    csv_writer.write_results('results/qwen/structured_greedy_shuffled.csv', [], headers=['clip','pred'])

if __name__ == '__main__':
    run()
