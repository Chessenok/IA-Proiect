import sys
from src.utils import manhattan


def main():
    print(f'automat(scipy) - {manhattan.calculateAutomaticallyFromFile()}')
    print(f'manual - {manhattan.calculateFromFile()}')

if __name__ == "__main__":
    sys.exit(main())