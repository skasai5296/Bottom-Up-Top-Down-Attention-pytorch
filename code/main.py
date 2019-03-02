import argparse



def main(args):
    print(hoge)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=100)
    args = parser.parse_args()
    main(args)
