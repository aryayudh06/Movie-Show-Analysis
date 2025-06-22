Sure! Here's your updated `README.md`, now reflecting that the JWT key should be added under the `&airflow-common-env` anchor in the `docker-compose.yml` file:

````markdown
# 🎬 Movie Show Analysis with Apache Airflow

This project aims to analyze movie and streaming show data using an **ETL (Extract, Transform, Load)** pipeline executed through **Apache Airflow**. The pipeline extracts data from sources, transforms it, and stores it for further analysis.

---

## 🛠️ Technologies Used

- Python  
- Apache Airflow  
- Docker & Docker Compose  
- Apache Kafka (Streaming Processing)  
- MongoDB (Database)  
- TMDB & TvMaze API  
- Streamlit  

---

## ⚙️ Prerequisites

Before running the project, make sure you have the following installed:

- [Docker](https://www.docker.com/)
- [Docker Compose](https://docs.docker.com/compose/)
- [Python 3.10+](https://www.python.org/downloads/)
- [MongoDB Community Edition](https://www.mongodb.com/try/download/community)

### Install MongoDB Locally

1. **Download MongoDB**:  
   Visit the [MongoDB Community Download Page](https://www.mongodb.com/try/download/community) and download the version for your operating system.

2. **Install MongoDB**:
   - **Windows**: Use the `.msi` installer and make sure "MongoDB as a Service" is checked.
   - **macOS**:
     ```bash
     brew tap mongodb/brew
     brew install mongodb-community@6.0
     ```
   - **Linux (Ubuntu)**:
     ```bash
     sudo apt update
     sudo apt install -y mongodb
     ```

3. **Start MongoDB Service and Connect to Connection**:
   - **macOS/Linux**:
     ```bash
     brew services start mongodb/brew/mongodb-community
     # or
     sudo service mongod start
     ```
   - **Windows**: MongoDB should start automatically as a service. You can verify via `services.msc`.

4. **Verify MongoDB is Running**:
   ```bash
   mongo
````

or

```bash
mongosh
```

---

## 🚀 How to Run the Project

### Additional Configuration (if using JWT Authentication in Airflow)

If your Airflow setup requires JWT for API authentication, add the following environment variable under the `&airflow-common-env` section in your `docker-compose.yml`:

```yaml
AIRFLOW__API_AUTH__JWT_SECRET: "my_super_secret_key"
AIRFLOW__API_AUTH__JWT_PRIVATE_KEY_PATH: ""
```

Example:

```yaml
x-airflow-common-env: &airflow-common-env
  AIRFLOW__CORE__EXECUTOR: LocalExecutor
  AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
  AIRFLOW__API__AUTH_BACKENDS: 'airflow.api.auth.backend.jwt_auth'
  AIRFLOW__API__AUTH_JWT_SECRET: "my_super_secret_key"
  AIRFLOW__API_AUTH__JWT_PRIVATE_KEY_PATH: ""
```

---

### Run Airflow

1. Start Airflow using Docker Compose:

   ```bash
   docker compose up
   ```

2. Open the Airflow UI in your browser:
   [http://localhost:8080](http://localhost:8080)

3. Log in using the default credentials (check your `docker-compose.yml`), then trigger the DAG named:

   ```
   streaming_elt_pipeline
   ```

4. To stop all services:

   ```bash
   docker compose down
   ```

   or

   ```bash
   docker stop <container-name>
   ```

---

### Run Model Training
1. Change to folder directory:

   ```bash
   cd Movie-Show-Analysis
   ```

2. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

3. change to virtual environment

   ```bash
   venv/Scripts/activate
   ```

4. Run the training pipeline:

   ```bash
   python Model/modelpipeline.py
   ```

5. If training pipeline error, run the code individually:

   ```bash
   python Model/DatasetImporter.py
   ```
   ```bash
   python Model/model.py
   ```

---

### Run Data Visualization (Dashboard)

1. Navigate to the dashboard folder:

   ```bash
   cd dashboard
   ```

2. Start the Streamlit app:

   ```bash
   streamlit run dashboard.py
   ```

---

## 📁 Project Structure

```bash
.
├── dags/                   # Airflow DAGs folder
│   └── ELT Scripts
├── docker-compose.yml      # Docker Compose configuration for Airflow
├── Model/                  # Model training pipeline
│   └── modelpipeline.py
├── dashboard/              # Streamlit dashboard for visualization
│   └── dashboard.py
├── requirements.txt
└── README.md
```

---

## ✅ Summary

This project integrates ETL pipelines, model training, and visualization into a unified environment using modern tools such as Apache Airflow, Docker, Kafka, MongoDB, and Streamlit for end-to-end data analysis of movies and streaming content.
