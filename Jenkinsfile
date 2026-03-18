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
                    sh '$HOME/.local/bin/uv run python src/prepareData.py'
            }
        }
    }
}