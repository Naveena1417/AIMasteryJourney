pipeline {
    agent any

    stages {

        stage('Build Docker Image') {
            steps {
                echo 'Building AI Research ML Crew Docker image...'
                sh 'docker build -t ai-research-ml-crew:jenkins .'
            }
        }

        stage('Run Tests') {
            steps {
                echo 'Running pytest inside Docker image...'
                sh 'docker run --rm --network ai-ml-network -e OLLAMA_HOST=http://host.docker.internal:11434 -e MLFLOW_TRACKING_URI=http://mlflow-server:5000 ai-research-ml-crew:jenkins python -m pytest -v'
            }
        }
    }

    post {
        success {
            echo 'AI Research ML Crew CI pipeline completed successfully!'
        }

        failure {
            echo 'Pipeline failed. Check the console output.'
        }
    }
}