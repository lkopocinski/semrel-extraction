import sys
import torch
import numpy as np

from relextr.model.scripts import RelNet

def prRed(prt):
	print("\033[91m {}\033[00m" .format(prt))


def prGreen(prt):
	print("\033[92m {}\033[00m" .format(prt))


def main():
	mapping = {
		0: 'no_relation',
		1: 'in_relation'
	}

	network = RelNet()
	network.load('../../model/relextr_model.pt')
	for line in sys.stdin:
		try:
			rel, v_a, v_b, w_a, w_b = line.strip().split('\t')
		except ValueError:
			try:
				rel, v_a, v_b, w_a, c_a, ctx_a, w_b, c_b, ctx_b, vc_a, vc_b = line.strip().split('\t')
			except ValueError:
				continue
		v_a = np.array(eval(v_a))
		v_b = np.array(eval(v_b))
		vc_a = np.array(eval(vc_a))
		vc_b = np.array(eval(vc_b))
		v = np.concatenate([v_a, v_b, vc_a, vc_b])
		#v = np.concatenate([vc_a, vc_b])


		output = network(torch.FloatTensor([v]))
		_, predicted = torch.max(output, dim=1)
		if mapping[predicted.item()] == rel:
			pass
			prGreen('\n{} : {}\tpred: {}\ttrue: {}\t{}\t{}'.format(w_a, w_b, mapping[predicted.item()], rel, ctx_a, ctx_b))
		else:
			prRed('\n{} : {}\tpred: {}\ttrue: {}\t{}\t{}'.format(w_a, w_b, mapping[predicted.item()], rel, ctx_a, ctx_b))


if __name__ == "__main__":
	main()
