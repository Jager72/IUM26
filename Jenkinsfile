pipeline {
    agent {
        dockerfile {
            filename 'Dockerfile'
            reuseNode true
        }
    }

    parameters {
        string(name: 'CUT_OFF', defaultValue: '', description: 'Limit dataset size')
    }

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
                script {
                    def cmd = '$HOME/.local/bin/uv run python src/prepareData.py'

                    if (params.CUT_OFF?.trim()) {
                        cmd += " --cut-off ${params.CUT_OFF}"
                    }

                    sh cmd
                }
            }
        }

        stage('Archive Artifact') {
            steps {
                archiveArtifacts artifacts: 'artifacts/splits.txt', fingerprint: true
            }
        }
    }
}