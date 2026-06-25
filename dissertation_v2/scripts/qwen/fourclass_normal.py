"""Placeholder for fourclass_greedy_normal experiment."""
from shared import prompts, csv_writer

def run():
    prompt = prompts.get_prompt('fourclass','greedy_normal')
    print('Running', prompt)
    csv_writer.write_results('results/qwen/fourclass_greedy_normal.csv', [], headers=['clip','pred'])

if __name__ == '__main__':
    run()
