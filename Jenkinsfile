pipeline {
    agent any

    environment {
        VENV_DIR = 'venv'
    }

    options {
        timestamps()
        disableConcurrentBuilds()
    }

    stages {

        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Set Up Python Environment') {
            steps {
                bat '''
                    python -m venv %VENV_DIR%
                    %VENV_DIR%\\Scripts\\python.exe -m pip install --upgrade pip
                    %VENV_DIR%\\Scripts\\python.exe -m pip install -r requirements.txt
                    %VENV_DIR%\\Scripts\\python.exe -m pip install requests
                '''
            }
        }

        stage('Train Model') {
            steps {
                bat '''
                    %VENV_DIR%\\Scripts\\python.exe train_model.py
                '''
            }
        }

        stage('Start API & Smoke Test') {
            steps {
                bat '''
                    start /B %VENV_DIR%\\Scripts\\python.exe app.py > app.log 2>&1

                    echo Waiting for API...
                    timeout /t 10 /nobreak

                    %VENV_DIR%\\Scripts\\python.exe test_prediction.py
                '''
            }
        }
    }

    post {
        always {
            archiveArtifacts artifacts: 'house_model.pkl, app.log',
                allowEmptyArchive: true

            bat '''
                taskkill /F /IM python.exe >nul 2>&1 || exit 0
            '''
        }

        success {
            echo 'Build, train, and smoke test succeeded.'
        }

        failure {
            echo 'Pipeline failed. Check Console Output and app.log.'
        }

        cleanup {
            bat '''
                if exist %VENV_DIR% rmdir /S /Q %VENV_DIR%
            '''
        }
    }
}
