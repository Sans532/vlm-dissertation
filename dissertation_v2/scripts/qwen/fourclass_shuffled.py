"""Placeholder for fourclass_greedy_shuffled experiment."""
from shared import prompts, csv_writer

def run():
    prompt = prompts.get_prompt('fourclass','greedy_shuffled')
    print('Running', prompt)
    csv_writer.write_results('results/qwen/fourclass_greedy_shuffled.csv', [], headers=['clip','pred'])

if __name__ == '__main__':
    run()
