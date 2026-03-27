pipeline {
    agent {
        docker {
            image 'jager72/ium:0.1.5'
            args '-u root'
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
        stage('Run Script') {
            steps {
                script {
                    def cmd = "uv run python src/prepareData.py"
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