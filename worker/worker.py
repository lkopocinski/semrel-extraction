import nlp_ws


class SemrelWorker(nlp_ws.NLPWorker):

    def process(self, input_path, task_options, output_path):
        if task_options.get('ner', False):
            # process with ner
        else:
            # process each




if __name__ == '__main__':
    print('Start')
    nlp_ws.NLPService.main(SemrelWorker)


