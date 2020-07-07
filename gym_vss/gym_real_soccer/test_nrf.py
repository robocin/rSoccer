from nrfparser import NRFParser
import argparse


def argument_parser():
	# Set args
	parser = argparse.ArgumentParser(description='Teste NRF')
	parser.add_argument('--id', type=int, default=0, help='Id')

	return parser.parse_args()


args = argument_parser()
ctrl = NRFParser()

while True:
	print("send speeds", args.id)
	ctrl.send_speeds(40, -40, args.id)
