1. Overview
This repository provides Python implementations for data preprocessing, clustering analysis, graph data preparation, model training, and experimental evaluation.

2. Project Structure
RT-DHGNN/

├── data/                            

├── result/                            

├── checkpoints/                      

│

├── README.md

├── requirements.txt

├── .gitignore

│

├── construct_role_lable.py

├── select_optimal_K.py

├── reconstruct_best_role_lable.py

├── Organization-Activity _Heatmap.py

├── build_rt_dhgnn_graph.py

├── RT-DHGNN-HGT.py

├── role_transition_incremental_gain.py

├── role_transition_information_gain.py

├── role_transition_shuffle_test.py

└── RT-GHGNN-Ablation.py

Generated directories are created during execution when required.



3. Environment Requirements

The project is implemented in Python and uses common data processing, machine learning, and deep learning libraries.

Install the dependencies listed in `requirements.txt`:

pip install -r requirements.txt

PyTorch and PyTorch Geometric are also required for graph-related scripts.

Install these packages separately according to your Python and CUDA environment.



4. Execution Instructions

4.1 Main Workflow

Data preprocessing: construct_role_lable.py

Parameter selection: select_optimal_K.py

Result reconstruction:  reconstruct_best_role_lable.py

Data visualization:  Organization-Activity _Heatmap.py

Graph data construction:  build_rt_dhgnn_graph.py

Model training and evaluation:  RT-DHGNN-HGT.py

4.2 Auxiliary Analysis

The following scripts perform additional analyses and require their corresponding input data files.

Incremental analysis: role_transition_incremental_gain.py

Information analysis: role_transition_information_gain.py

Shuffle validation: role_transition_shuffle_test.py

4.3  Comparative Experiments

Ablation experiments: RT-GHGNN-Ablation.py



5.Output Files

Depending on the executed scripts, the following types of files may be generated:

- Processed CSV files

- Graph data files

- Intermediate metadata files

- Model checkpoints

- Evaluation tables

- Visualization figures

Output locations are controlled by the configuration settings in each script.



6. Reproducibility Notes

- Prepare all required input files before execution.

- Check the configured input and output paths.

- Follow the specified execution order for dependent scripts.

- Auxiliary analyses may be executed independently when their input files are available.

- Use compatible versions of PyTorch and PyTorch Geometric.

- GPU acceleration is recommended for computationally intensive tasks.



7.dataset

Event logs for predictive business process monitoring can be obtained from 4TU Research Data.



