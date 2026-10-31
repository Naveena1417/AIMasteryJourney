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
                sh 'docker run --rm ai-research-ml-crew:jenkins python -m pytest -v'
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