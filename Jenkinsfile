pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Install uv') {
            steps {
                sh 'curl -Ls https://astral.sh/uv/install.sh | sh'
            }
        }

        stage('Setup Environment') {
            steps {
                sh '$HOME/.local/bin/uv sync'
            }
        }

        stage('Run Script') {
            steps {
                withCredentials([string(credentialsId: 'KAGGLE_ENV', variable: 'KAGGLE_API_TOKEN')]) {
                    sh 'env | grep KAGGLE || true'
                    sh '$HOME/.local/bin/uv run python main.py'
                }
            }
        }
    }
}