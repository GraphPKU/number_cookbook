# Number Cookbook: Number Understanding of Language Models and How to Improve It

## Introduction
Welcom to the official code repository for the paper [Number Cookbook: Number Understanding of Language Models and How to Improve It](https://arxiv.org/abs/2411.03766). In this work, we introduce a novel benchmark that focuses on the **Number Understanding and Processing Ability** (**NUPA**) of language models. The benchmark encompasses four distinct number *representations* and spans 17 diverse *tasks* that test number understanding and processing capabilities. 

We evaluate various language models across different input length ranges and performance metrics. Additionally, we explore how different tokenization strategies, positional encodings (PEs), and data formats impact NUPA. The effectiveness of the Chain of Thought (CoT) approach is also assessed on our benchmark to provide further insights into improving numerical reasoning in language models.

## Benchmark Overview
Our benchmark consists of 4 number representations:
  - Integer: e.g., 123
  - Float: e.g., 123.45
  - Fraction: e.g., 12/5
  - Scientific Notation: e.g., 1.23e2

And 17 number understanding and processing tasks:

<table>
  <tr>
    <th rowspan="2">Representation</th>
    <th colspan="6">Elementary Arithmetic</th>
    <th colspan="2">Comparison</th>
    <th colspan="6">Digit Understanding</th>
    <th colspan="3">Conversion</th>
  </tr>
  <tr>
    <th>Add</th>
    <th>Sub</th>
    <th>Multiply</th>
    <th>Truediv</th>
    <th>Floordiv</th>
    <th>Mod</th>
    <th>Max</th>
    <th>Min</th>
    <th>Digit Max</th>
    <th>Digit Min</th>
    <th>Digit Add</th>
    <th>Get Digit</th>
    <th>Length</th>
    <th>Count</th>
    <th>To Float</th>
    <th>To Scientific</th>
    <th>Sig. Fig.</th>
  </tr>
  <tr>
    <td><strong>Integer</strong></td>
    <td>✔</td>
    <td>✔</td>
    <td>✔</td>
    <td>✔</td>
    <td>✔</td>
    <td>✔</td>
    <td>✔</td>
    <td>✔</td>
    <td>✔</td>
    <td>✔</td>
    <td>✔</td>
    <td>✔</td>
    <td>✔</td>
    <td>✔</td>
    <td>—</td>
    <td>✔</td>
    <td>✔</td>
  </tr>
  <tr>
    <td><strong>Float</strong></td>
    <td>✔</td>
    <td>✔</td>
    <td>✔</td>
    <td>✘</td>
    <td>—</td>
    <td>—</td>
    <td>✔</td>
    <td>✔</td>
    <td>✔</td>
    <td>✔</td>
    <td>✔</td>
    <td>✔</td>
    <td>✔</td>
    <td>○</td>
    <td>—</td>
    <td>✔</td>
    <td>✔</td>
  </tr>
  <tr>
    <td><strong>Fraction</strong></td>
    <td>✔</td>
    <td>✔</td>
    <td>✔</td>
    <td>✔</td>
    <td>—</td>
    <td>—</td>
    <td>✔</td>
    <td>✔</td>
    <td>—</td>
    <td>—</td>
    <td>—</td>
    <td>—</td>
    <td>—</td>
    <td>○</td>
    <td>✔</td>
    <td>○</td>
    <td>○</td>
  </tr>
  <tr>
    <td><strong>Scientific</strong></td>
    <td>✔</td>
    <td>✔</td>
    <td>✔</td>
    <td>✘</td>
    <td>—</td>
    <td>—</td>
    <td>✔</td>
    <td>✔</td>
    <td>—</td>
    <td>—</td>
    <td>—</td>
    <td>—</td>
    <td>—</td>
    <td>○</td>
    <td>✔</td>
    <td>—</td>
    <td>○</td>
  </tr>
</table>

The details about the tasks can be found in our [paper](https://arxiv.org/abs/2411.03766).

### Leaderboard

We have evaluated several popular LLMs on our benchmark and the results are summarzied in the [leaderboard](https://huggingface.co/spaces/KangShijia/NUPA-Leaderboard). This leaderboard will be regularly updated as new evaluations are conducted.

We welcome contributions from the community! If you have suggestions for additional model evaluations or wish to share your own results, feel free to reach out by proposing an issue or emailing us at <haotongyang@pku.edu.cn>.

### Download the benchmark data
We offer two versions of our benchmark: 
  1. [number dataset](xxxxxxxx):  
    - A binary file containing the number pairs to conduct the number understanding and processing tasks. Numbers are represented using our custom number class.  
    - Ideal for training models from scratch without textual context or for experimenting with custom text formulations.
  
  2. [text dataset](xxxxxxxx):  
    - This version converts numbers and tasks into text Q&A pairs.  
    - Suitable for testing or fine-tuning models on text data.

**Note**: The number dataset consists of number pairs, enabling shared usage of numbers across various tasks. Since different tasks have unique input requirements, these pairs need further preprocessing using the `Task.preprocess` method.

To simplify this process, you can load the files using `number_model.dataset.NumberDataset`, which automatically preprocesses the number pairs into the specified tasks.

## Installation
First, clone the repository and install the required packages by running the following command:

```bash
conda create -n nupa python=3.11
conda activate nupa
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Create the benchmark data
If you wish to generate the benchmark data from scratch, follow the instructions below.

**Note**: This step is *optional*. If your goal is solely to test or fine-tune models on the benchmark, you can skip this process and directly download the pre-generated benchmark data from the provided [links](xxxxxxxx).

### Create your own number dataset
To create your own number dataset, use the script `src/datagen/data_generate.py`. For example, the following command generates a dataset of `Integer` pairs:
```bash
python -m src.datagen.data_generate -d Integer -n 10000 --min_len 1 --max_len 10 --min_valid_len 3 --max_valid_len 20 --valid_nums 1000 --min_test_len 3 --max_test_len 20 --test_nums 1000 --save_path ./data
```
This command will generate 10000 `Integer` pairs for training, where the training data has a length (digit) range from 1 to 10 with 10000 samples for each length, and the validation data and test data have a length range from 3 to 20 with 1000 samples for each length. You can change the number representation to `Float`, `Fraction`, or `ScientificNotation` by adjusting the `-d` parameter. Three binary files `train.pkl`, `valid.pkl` and `test.pkl` will be generated in the `save_path` directory. There are a list of pairs of numbers in given class (representation) in these binary files.

For more details, run 
```bash
python -m src.datagen.data_generate -h
```

To create the number dataset we use to further generate the text dataset, you can run the following command:
  
  ```bash
  python -m src.datagen.benchmark_generate -d --decon -c
  ```

These number datasets will be saved in the `benchmark/numbers` directory.

### Create your own text dataset
To create the text dataset, run the following command:

```bash
python -m src.datagen.benchmark_generate -d -t --decon -c
```
Here's what each flag does:
  - `-d`: Generate the number datasets and save them in the `benchmark/numbers` directory.
  - `-t`: Converts the generated number datasets into a text dataset, saved as JSON files in the `benchmark/tasks` directory.
  - `--decon`: Further decontainment of the number dataset to make sure no data leakage. These containment could be introduced by the task-specific preprocess of the number dataset.
  - `-c`: Skip generation for datasets and tasks if exists.

These step could consume about several hours. The `src/datagen/benchmark_config.py` file record the needed number datasets and the mapping to the tasks. 

### Create your own Rule-Following CoT dataset
To create your own RF-Cot dataset, use the script `src/datagen/data_generate.py` to convert your text dataset into RF-Cot format. 

```bash
python -m src.datagen.rfft.generate --model_path <model_path>
```
This command transforms `benchmark/tasks/{train/valid/test}.json` into `benchmark/tasks/rfft_{train/valid/test}.json`.

## Test on the benchmark
### Test huggingface models on the benchmark
Once you have the benchmark data, you can test any Hugging Face model huggingface by running:

```bash
python -m src.text_model.benchmark_test <model_name_or_path>
```

This will evaluate the model on the benchmark, and the raw generation results will be saved in a JSON file in the `benchmark/test_results/` directory. 

After generating results for all models,  you can create a performance report by running:

```bash
python -m src.text_model.readout_generation --load_path <path_to_raw_results> --save_path <report_path>
```
The report will be saved as a text file in the benchmark/test_results_report/ directory. The report will be saved in the `save_path` directory as a txt file.

### Test OpenAI models on the benchmark
You can also test OpenAI models via their API with the following command:

```bash
python -m src.text_model.benchmark_test <model_name> --openai
```
This script uses [OpenAI batch API](https://platform.openai.com/docs/guides/batch) requests to minimize costs. Ensure your OPENAI_API_KEY is set in your environment variables. The request IDs will be stored in the `benchmark/gpt_request_report/` directory.

To check the status of your requests and download the results, run:
```bash
python -m src.text_model.download_from_gpt <model_name>
```
It may take from a few hours to a day to complete the requests. Once the results are downloaded, you can generate a report using the same `readout_generation` script.

## Fine-tune on the benchmark
To fine-tune a model on the benchmark, use the following command:

```bash
python -m src.text_model.benchmark_train --model_path <model_path> --dataset_path <dataset_path>
```

If you plan to fine-tune the model using custom positional encodings (PE), tokenizers, or data formats, refer to `src/text_model/benchmark_train.py` for details. 

**Note**: if you modify the data format (such as reverse representation or zero-padding), you must regenerate a corresponding text dataset provide its path. For example, to finetune a model with reversed number representation, please generate the text dataset:
```bash
python -m src.datagen.benchmark_generate -d -t --decon -c --reverse int
```
Once fine-tuning is complete, you can test a speicific checkpoint with:

```bash
python -m src.text_model.benchmark_test <model_name_or_path> --checkpoint_path <checkpoint_path>
```

Please make sure that the setting (tokenizer, pe, data format) of the checkpoint is the same as the setting of the finetuning. Details see `python -m src.text_model.benchmark_test -h`.

### Fine-tune with RF-CoT
To fine-tune models with RF-CoT, use the following command:

```bash
python -m src.text_model.rfft_train --model_path <model_path> --dataset_path <dataset_path>
```

To test models with RF-CoT, add the --rfft parameter to the above commands. For example:

```bash
python -m src.text_model.benchmark_test <model_name_or_path> --checkpoint_path <checkpoint_path> --rfft
```

## Training a number model from scratch
We also provide a script to train a number model from scratch, implementing various techniques, including custom PEs, tokenizers, and data formats.

To train a model on a single task from the benchmark:

```bash
python -m src.number_model.train -d <number_representation> -m <model_config_path> -p <number_dataset_path> -t <task_name>
```

To train a model on multiple tasks:
  
```bash
python -m src.number_model.train -m <model_config_path> --multi_task_config <path_to_config>
```

The configuration file should be a JSON containing a list of tuples: `(representation_name, task_name, dataset_path)`.

**Tracking**: We use TensorBoard to record the training and eval results.

For additional options and techniques, see python -m src.number_model.train -h or refer to src/number_model/train.py.

## Citation
Haotong Yang, Yi Hu, Shijia Kang, Zhouchen Lin, Muhan Zhang. *Number Cookbook: Number Understanding of Language Models and How to Improve It.* arXiv preprint arXiv:2411.03766, 2024.

Bibtex: 
```
@misc{yang2024numbercookbooknumberunderstanding,
      title={Number Cookbook: Number Understanding of Language Models and How to Improve It}, 
      author={Haotong Yang and Yi Hu and Shijia Kang and Zhouchen Lin and Muhan Zhang},
      year={2024},
      eprint={2411.03766},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.03766}, 
}
```
