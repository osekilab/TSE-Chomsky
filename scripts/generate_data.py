import fire
from chomsky_neural.data.generate_data import (
    generate_AnB_data,
    generate_AnBn_data,
    generate_dependency_data,
)

if __name__ == "__main__":
    fire.Fire(
        {
            "an_b": generate_AnB_data,
            "an_bn": generate_AnBn_data,
            "dependency": generate_dependency_data,
        }
    )
