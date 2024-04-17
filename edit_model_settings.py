import json
import sys
import os

def update_model_configuration(model_type):

    config_file_path = "model-config.json"
    model_config = {
        "name": "uplift-predictor",
        "implementation": "inference.UpliftModel",
        "parameters": {"uri": ""}
    }

    if model_type == "solo":
        model_config["parameters"]["uri"] = "one_model.joblib"
    elif model_type == "two":
        model_config["parameters"]["uri"] = "two_model.joblib"
    else:
        raise ValueError("Invalid model type.")

    try:
        os.remove(config_file_path)
    except FileNotFoundError:
        print(f"{config_file_path} was not found.")

    with open(config_file_path, "w") as file:
        json.dump(model_config, file, indent=4)
    print(f"configs updated successfully in {config_file_path}.")

def process_command_line_arguments():
    if len(sys.argv) != 2 or sys.argv[1] not in ["solo", "two"]:
        print("Error: need an argument from a list")
        sys.exit(-1)
    return sys.argv[1]

def main():
    model_type = process_command_line_arguments()
    update_model_configuration(model_type)

if __name__ == "__main__":
    main()
