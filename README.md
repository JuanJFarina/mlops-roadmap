# MLOps Roadmap

Complete roadmap to become a Senior MLOps. It covers Software Engineering, Backend Development, Data Analyst / Scientist / Engineer, DevOps, MLOPs and LLMOps concepts as well as books to learn about them in depth.

## Concepts grouped by roles

### Software Engineering
- **Math**
    - Elementary Algebra
        - Linear Algebra
        - Matrix Operations
        - Vector Spaces
        - Eigenvalues and Eigenvectors
    - Probability & Statistics
        - Mean, Median and Mode
        - Quartiles and Deciles
        - Correlation and Causation
        - Variance and Covariance
        - Standard Deviation
        - Bayes Theorem
        - Gaussian Distributions
        - Hypothesis Testing
        - Markov Chains
        - Confidence Intervals
    - Calculus
        - Differentiation
        - Partial Derivatives
        - Gradient Descent
    - Discrete Math
        - Sets
        - Relations
        - Combinatorics
        - Graph Theory
- **Logic**
    - Propositional Logic
        - Truth Tables
        - Logical Equivalences
        - Normal Forms (DNF, CNF)
        - Inference Rules
    - Boolean Algebra
        - Logic Gates
        - Simplifying Circuits
        - Truth Minimization
    - Predicate Logic
        - Basic Concepts
        - Quantifiers
        - First-Order Logic
- **Computation**
    - Numeral Systems
        - Binary
        - Hexaecimal
    - Computer Architecture
        - Central Processing Unit
        - Read Only Memory
        - Random Access Memory
        - Central Bus System
    - Networks
        - Topology
        - OSI Model
        - TCP/IP Model
            - Ethernet
            - IP
            - TCP, FTP
            - HTTP, HTTPS
    - Computational Problems
- **Data Structures**
    - Arrays, Lists, Dictionaries
    - Queues, Stacks
    - Sets, Hash Maps
    - Trees
    - Graphs
    - Heaps
    - Bloom Filters
- **Algorithms**
    - Asymptotic Analysis (Big O notation)
    - Divide & Conquer
    - Sorting
    - Searching
    - Graph
        - DFS
        - BFS
        - Minimum Spanning Tree
    - Dynamic Programming
        - Memoization
        - Tabulation
    - Greedy Algorithms
    - Backtracking
- **Coding**
    - Low-Level Structured Programming
        - Memory allocation
        - Variables and pointers
        - Garbage collection
        - Compilation
    - High-Level Structured Programming
    - Clean Code
    - Object-Oriented Programming
    - SOLID
    - Functional Programming
    - KISS, YAGNI
    - Git concepts and commands
    - Git workflow strategies
    - Debugging Tools
    - Advanced Concepts
        - Synchronous vs Asynchronous
        - Multithreading and Race conditions
- **Software Design**
    - Quality Metrics
        - Coupling
        - Cohesion
        - Instability
        - Cyclomatic Complexity
        - Logical Lines of Code
        - Cognitive Complexity
    - Design Patterns
        - Dependency Injection
        - Dependency Inversion
        - Singleton
        - Factory
        - Builder
        - Adapter
        - Facade
        - Repository
        - Observer
        - Strategy
- **Software Architecture**
    - Transaction Script
    - Active Record
    - Domain Model
    - Layered Architecture
    - N-Tier Architecture
    - Hexagonal Architecture (Clean Code or Ports & Adapters)
    - CQRS
- **Testing**
    - Types
    - Strategies
    - White-Box Testing
    - Chicago TDD
    - London TDD
- **Software Development Life-Cycle**
    - Waterfall
    - Extreme Programming
    - Lean Start-Up
    - Scrum

-------------------------------------------------------

### Backend Development
- **API Design**
    - REST
    - gRPC
    - GraphQL
    - Versioning
    - Authentication
        - JSON Web Token (JWT)
        - OAuth 2.0
    - Security
        SQL Injections
    - Rate Limiting
    - Documentation
    - Idempotency
    - Rate Limiting
    - Concurrency
    - Caching
- **Managers, Frameworks & Libraries (Python Specific)**
    - Pip & Pipenv
    - Pydantic
    - Pytest
    - FastAPI / Django / Flask
- **Databases**
    - Normalization
    - Denormalization
    - Indexing
    - Transaction
    - Query Optimization
    - Data Partitioning
    - Data Sharding
    - Replication Strategies
    - SQL
        - SELECT FROM queries
        - INSERT INTO queries
        - Filter operators
        - Join operators
    - Object-Relational Mapping (ORM)
    - NoSQL Databases and Commands
- **Cloud**
    - Microsoft Azure / GCP / AWS
    - Horizontal and Vertical Scaling
    - CAP Theorem
- **Messaging**
    - Kafka
    - RabbitMQ

-------------------------------------------------------

### DevOps
- **Containerization & Orchestration**
    - Docker
        - Docker networking
        - Docker-Compose
    - Kubernetes
    - ArgoCD
    - KubeFlow
- **CI/CD**
    - Bash & Powershell
    - Linux Commands
    - Azure Pipelines / Github Actions
    - SonarQube
    - Veracode
    - Jenkins
    - CircleCI
- **Infrastructure**
    - Terraform
    - Ansible
    - AWS CloudFormation
    - Azure Resource Manager
    - Service Discovery
    - Load Balancing
    - Rolling updates
    - Immutable infrastructure
- **Deployment Strategies**
    - Canary releases
    - Blue-Green deployments
- **Secrets Management**
- **Monitoring**
    - Logging
    - Metrics
        - SLA, SLO, SLI
        - DataDog
        - Azure Log Analytics
        - NewRelic
    - Distributed tracing
        - OpenTelemetry
        - Prometheus
        - Grafana
        - ELK Stack

-------------------------------------------------------

### Data Analysis
- **Structured Data Formats**
    - Comma Separated Values (CSV)
    - Tab Separated Values (TSV)
    - Parquet
    - Time Series
- **Unstructured Data Formats**
- **Exploratory Data Analysis (EDA) & Data Cleaning**
    - Jupyter Notebooks
    - Numpy
    - Pandas
        - Missing values
        - Outliers
- **SQL Commands**
    - SELECT FROM queries
    - INSERT INTO queries
    - Filter operators
    - Join operators
- **Data Visualization**
    - Matplotlib
    - Seaborn
    - Streamlit / Chainlit
    - Tableau
    - PowerBI

-------------------------------------------------------

### Data Engineering
- **Data Storages**
    - Data Lakes
    - Data Warehouses
- **Extract-Transform-Load (ETL)**
    - Feature Engineering
    - Polynomial Features
    - One-Shot Encoding
    - Embedding Generation
    - Categorical Variables
- **Data Pipelines & Processing**
    - Data Schema Evolution
    - Columnar Databases
    - Batch processing
    - Stream processing
    - Apache Airflow
    - Apache Spark
    - Apache Flink

-------------------------------------------------------

### Data Science
- **Supervised Algorithms**
    - Classification vs Regression
    - Decision Trees
    - Gradient Descent
    - Linear Regression
    - Logistic Regression
    - Random Forest
    - Gradient Boosting
- **Unsupervised Algorithms**
    - Clustering
    - K-Means
    - K-Nearest Neighbors
    - Principal Component Analysis (PCA)
    - Content-Based Recommendation Systems
    - Collaborative Filtering
    - Hybrid Recommendation Systems
- **Deep Learning**
    - Artificial Neural Networks (ANNs)
    - Convolutional Neural Networks (CNNs)
    - Recurrent Neural Networks (RNNs)
    - Transformer Networks
    - Generative Adversarial Networks (GANs)
- **Reinforcement Learning**
- **Libraries**
    - Scikit-Learn
    - TensorFlow, Keras
    - PyTorch
    - XGBoost
- **Model Optimization**
    - Imbalanced datasets
    - Bias-variance tradeoff
    - Hyperparameter Tuning
    - Regularization
    - Cross-validation
    - Principal Component Analysis (PCA)
- **A/B Testing and Evaluation Metrics**
    - Accuracy
    - F-1 Score
    - Confusion Matrix
    - Precision vs Recall
    - Precision-Recall Curves
    - Receiver Operating Characteristic (ROC) Curves
    - Area Under the Curve (AUC)

-------------------------------------------------------

### MLOps
- **Platforms**
    - Cloud-Native ML Services
    - MLFLow
    - Kubeflow
    - Sagemaker Pipelines
- **Orchestration**
    - ML Pipelines
    - Online Inference
    - Batch Inference
- **Feature Stores**
- **Data Versioning**
- **Model Versioning**
    - Metadata Management
    - Model Registry
    - Experiment Tracking
- **Monitoring & Observability**
    - Model Drift Detection
    - Fairness Audits
    - A/B Testing

-------------------------------------------------------

### LLMOps
- **Training**
- **Inference**
- **Fine-Tuning**
    - Adapters (LoRA)
    - Parameter Efficient Fine-Tuning (PEFT)
    - Domain-Specific Pretraining
- **Optimization**
    - Quantization
    - Pruning
    - Latency Optimization
    - Throughput
    - Hallucinations
    - Ethical concerns and biases
    - Prompt Engineering
- **LLMs**
    - ChatGPT
    - Llama
    - Mistral
    - Bert
- **LLM Design Patterns**
    - Retrieval-Augmented Generation (RAG)
- **Frameworks**
    - LangChain
    - Autogen
    - PydanticAI

## Textbooks

1. **"Introduction to the Theory of Computation"** by Michael Sipser
1. **"Structure and Interpretation of Computer Programs"** by Harold Abelson and Gerald Jay Sussman
1. **"Introduction to Algorithms"** by Thomas H. Cormen, Charles E. Leiserson, Ronald L. Rivest, and Clifford Stein
1. **"Design Patterns: Elements of Reusable Object-Oriented Software"** by Erich Gamma, Richard Helm, Ralph Johnson, and John Vlissides
1. **"Computer Organization and Design: The Hardware/Software Interface"** by David A. Patterson and John L. Hennessy
1. **"Operating System Concepts"** by Abraham Silberschatz, Peter B. Galvin, and Greg Gagne
1. **"Computer Networking: A Top-Down Approach"** by James F. Kurose and Keith W. Ross
1. **"Database System Concepts"** by Abraham Silberschatz, Henry Korth, and S. Sudarshan
1. **"Readings in Database Systems"** edited by Joseph M. Hellerstein and Michael Stonebraker
1. **"Artificial Intelligence: A Modern Approach"** by Stuart Russell and Peter Norvig
1. **"Pattern Recognition and Machine Learning"** by Christopher M. Bishop
1. **"Concrete Mathematics: A Foundation for Computer Science"** by Ronald L. Graham, Donald E. Knuth, and Oren Patashnik
1. **"Discrete Mathematics and Its Applications"** by Kenneth H. Rosen
1. **"Automata Theory, Languages, and Computation"** by John E. Hopcroft, Rajeev Motwani, and Jeffrey D. Ullman
1. **"Cryptography and Network Security: Principles and Practice"** by William Stallings
1. **"Applied Cryptography: Protocols, Algorithms, and Source Code in C"** by Bruce Schneier
1. **"The Elements of Statistical Learning"** by Trevor Hastie, Robert Tibshirani, and Jerome Friedman
1. **"Data Mining: Concepts and Techniques"** by Jiawei Han, Micheline Kamber, and Jian Pei
