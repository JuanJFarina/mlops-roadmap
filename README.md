# MLOps Roadmap

Complete roadmap to become a Senior MLOps. It covers Software Engineering, Backend Development, Data Analyst / Scientist / Engineer, DevOps, MLOPs and LLMOps concepts as well as books to learn about them in depth.

## Concepts grouped by roles

### 1) SOFTWARE ENGINEERING
- **1.1) Math**
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
        - Statistical Significance and P-Value
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
- **1.2) Logic**
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
- **1.3) Computation**
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
- **1.4) Data Structures**
    - Arrays, Linked Lists, Dictionaries
    - Queues, Stacks
    - Sets, Hash Maps
    - Trees
    - Graphs
    - Heaps
    - Bloom Filters
- **1.5) Algorithms**
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
- **1.6) Coding**
    - Low-Level Structured Programming
        - Memory allocation
        - Variables and pointers
        - Garbage collection
        - Compilation
    - High-Level Structured Programming
    - Clean Code
    - Object-Oriented Programming
        - Encapsulation
        - Inheritance
        - Polymorphism
        - Abstraction
        - Method Overloading
        - Method Overriding
        - Interfaces
        - Abstract Classes
        - Composition
        - Aggregation
        - Messaging
    - SOLID
    - Functional Programming
    - KISS, YAGNI
    - Git concepts and commands
    - Git workflow strategies
    - Debugging Tools
    - Advanced Concepts
        - Synchronous vs Asynchronous
        - Multithreading and Race conditions
- **1.7) Software Design**
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
- **1.8) Software Architecture**
    - Transaction Script
    - Active Record
    - Domain Model
    - Layered Architecture
    - N-Tier Architecture
    - Hexagonal Architecture (Clean Code or Ports & Adapters)
    - CQRS
- **1.9) Testing**
    - Types
    - Strategies
    - White-Box Testing
    - Chicago TDD
    - London TDD
- **1.10) Software Development Life-Cycle**
    - Waterfall
    - Extreme Programming
    - Lean Start-Up
    - Scrum

-------------------------------------------------------

### 2) BACKEND DEVELOPMENT
- **2.1) API Design**
    - HTTP Requests and Responses
    - REST
    - gRPC
    - GraphQL
    - Versioning
    - Authentication
        - JSON Web Token (JWT)
        - OAuth 2.0
    - Security
        - SQL Injections
        - XSS
        - CSRF
        - DOS
        - DDOS
        - Brute Force
    - Rate Limiting
    - Documentation
    - Idempotency
    - Rate Limiting
    - Concurrency
    - Caching
- **2.2) Managers, Frameworks & Libraries (Python Specific)**
    - Pip & Pipenv
    - Pydantic
    - Pytest
    - FastAPI / Django / Flask
- **2.3) Databases**
    - ACID
    - Eventual Consistency
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
- **2.4) Cloud**
    - Microsoft Azure / GCP / AWS
    - Horizontal and Vertical Scaling
    - CAP Theorem
- **2.5) Messaging**
    - Kafka
    - RabbitMQ

-------------------------------------------------------

### 3) DEVOPS
- **3.1) Containerization & Orchestration**
    - Docker
        - Docker networking
        - Docker-Compose
    - Kubernetes
    - ArgoCD
    - KubeFlow
- **3.2) CI/CD**
    - Bash & Powershell
    - Linux Commands
    - Azure Pipelines / Github Actions / Jenkins
    - SonarQube
    - Veracode
    - Jenkins
    - CircleCI
- **3.3) Infrastructure**
    - Terraform
    - Ansible
    - AWS CloudFormation
    - Azure Resource Manager
    - Service Discovery
    - Load Balancing
    - Rolling updates
    - Immutable infrastructure
    - GitOps
- **3.4) Deployment Strategies**
    - Canary releases
    - Blue-Green deployments
- **3.5) Secrets Management**
- **3.6) Monitoring**
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

### 4) DATA ANALYSIS
- **4.1) Structured Data Formats**
    - Comma Separated Values (CSV)
    - Tab Separated Values (TSV)
    - Parquet
    - Time Series
- **4.2) Unstructured Data Formats**
- **4.3) Exploratory Data Analysis (EDA) & Data Cleaning**
    - Jupyter Notebooks
    - Numpy
    - Pandas
        - Duplicates
        - Missing values
        - Outliers
        - Pivot Tables
        - Groupby
        - Merge
- **4.4) SQL Commands**
    - SELECT FROM queries
    - INSERT INTO queries
    - Filter operators
    - Join operators
- **4.5) Data Visualization**
    - Matplotlib
    - Seaborn
    - Streamlit / Chainlit
    - Tableau
    - PowerBI

-------------------------------------------------------

### 5) DATA ENGINEERING
- **5.1) Data Storages**
    - Data Lakes
    - Data Warehouses
- **5.2) Extract-Transform-Load (ETL)**
    - Feature Engineering
    - Polynomial Features
    - One-Shot Encoding
    - Embedding Generation
    - Categorical Variables
- **5.3) Data Pipelines & Processing**
    - Data Schema Evolution
    - Columnar Databases
    - Batch processing
    - Stream processing
    - Apache Airflow
    - Apache Spark
    - Apache Flink

-------------------------------------------------------

### 6) DATA SCIENCE
- **6.1) Supervised Algorithms**
    - Classification vs Regression
    - Decision Trees
    - Gradient Descent
    - Linear Regression
    - Logistic Regression
    - Random Forest
    - Gradient Boosting
- **6.2) Unsupervised Algorithms**
    - Clustering
    - K-Means
    - K-Nearest Neighbors
    - Principal Component Analysis (PCA)
    - Content-Based Recommendation Systems
    - Collaborative Filtering
    - Hybrid Recommendation Systems
- **6.3) Deep Learning**
    - Artificial Neural Networks (ANNs)
    - Convolutional Neural Networks (CNNs)
    - Recurrent Neural Networks (RNNs)
    - Transformer Networks
    - Generative Adversarial Networks (GANs)
- **6.4) Reinforcement Learning**
- **6.5) Libraries**
    - Scikit-Learn
    - TensorFlow, Keras
    - PyTorch
    - XGBoost
- **6.6) Model Optimization**
    - Imbalanced datasets
    - Bias-variance tradeoff
    - Hyperparameter Tuning
    - Regularization
    - K-fold Cross-validation
    - Principal Component Analysis (PCA)
- **6.7) A/B Testing and Evaluation Metrics**
    - Accuracy
    - F-1 Score
    - Confusion Matrix
    - Precision vs Recall
    - Precision-Recall Curves
    - Receiver Operating Characteristic (ROC) Curves
    - Area Under the Curve (AUC)

-------------------------------------------------------

### 7) MLOPS
- **7.1) Platforms**
    - Cloud-Native ML Services
    - MLFLow
    - Kubeflow
    - Sagemaker Pipelines
- **7.2) Orchestration**
    - ML Pipelines
    - Online Inference
    - Batch Inference
- **7.3) Feature Stores**
- **7.4) Data Versioning**
- **7.5) Model Versioning**
    - Metadata Management
    - Model Registry
    - Experiment Tracking
- **7.6) Monitoring & Observability**
    - Model Drift Detection
    - Fairness Audits
    - A/B Testing

-------------------------------------------------------

### 8) LLMOPS
- **8.1) Training**
- **8.2) Inference**
- **8.3) Metrics**
    - BLEU
    - ROUGE
- **8.4) Fine-Tuning**
    - Adapters (LoRA)
    - Parameter Efficient Fine-Tuning (PEFT)
    - Domain-Specific Pretraining
- **8.5) Optimization**
    - Quantization
    - Pruning
    - Latency Optimization
    - Throughput
    - Hallucinations
    - Ethical concerns and biases
    - Prompt Engineering
    - Caching prompts-responses
- **8.6) LLMs**
    - ChatGPT
    - Llama
    - Mistral
    - Bert
- **8.7) LLM Design Patterns**
    - Retrieval-Augmented Generation (RAG)
- **8.8) Frameworks & Libraries**
    - LangChain
    - Autogen
    - PydanticAI
    - Hugging Face Transformers

## Textbooks

1. **Architectural Styles and the Design of Network-based Software Architectures** *by Roy Thomas Fielding* **[1.8, 2.1]**
1. **Clean Architecture: A Craftsman's Guide to Software Structure and Design** *by Robert C. Martin* **[1.6, 1.7, 1.8, 2.3]**
1. **Clean Code: A Handbook of Agile Software Craftsmanship** *by Robert C. Martin* **[1.4, 1.6, 1.7, 1.9]**
1. **CGI Programming in Perl** *by Kirrily Robert* **[2.1]**
1. **Core PHP Programming** *by Leon Atkinson* **[1.4, 1.5, 1.6, 2.1]**
1. **Cracking the Coding Interview** *by Gayle Laakmann McDowell* **[1.4, 1.5, 1.6]**
1. **Designing Data-Intensive Applications: The Big Ideas Behind Reliable, Scalable, and Maintainable Systems** *by Martin Kleppmann* **[1.4, 1.8, 2.1, 2.3, 4.1, 4.2, 5.1]**

-------------------------------------------------------

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
