Sure! Here's the complete English version of your updated `README.md` file, formatted properly for a GitHub README:

````markdown
# 🎬 Movie Show Analysis with Apache Airflow

This project aims to analyze movie and streaming show data using an **ETL (Extract, Transform, Load)** pipeline executed through **Apache Airflow**. The pipeline extracts data from sources, transforms it, and stores it for further analysis.

## 🛠️ Technologies Used

- Python  
- Apache Airflow  
- Docker & Docker Compose  
- Apache Kafka (Streaming Processing)  
- TMDB & TvMaze API  
- Streamlit  

## 🚀 How to Run the Project

Make sure you have [Docker](https://www.docker.com/) and [Docker Compose](https://docs.docker.com/compose/) installed.

### Additional Configuration (if using JWT Authentication in Airflow)

If your Airflow setup requires JWT for API authentication, add the following environment variable to your `docker-compose.yml`:

```yaml
AIRFLOW_API_AUTH_JWT_SECRET: my_super_secret_key
````

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

1. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the training pipeline:

   ```bash
   python Model/modelpipeline.py
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

This project integrates ETL pipelines, model training, and visualization into a unified environment using modern tools such as Apache Airflow, Docker, Kafka, and Streamlit for end-to-end data analysis of movies and streaming content.

```

Let me know if you'd like me to add a badge section, license, contributors, or GitHub actions integration!
```
