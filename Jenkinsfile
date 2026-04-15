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
        stage('Run Preparation') {
            steps {
                sh "mkdir -p artifacts"
                
                script {
                    def cmd = "uv run python src/prepareData.py"
                    if (params.CUT_OFF?.trim()) {
                        cmd += " --cut-off ${params.CUT_OFF}"
                    }
                    sh cmd
                }
            }
        }

        stage('Archive Prepared Data') {
            steps {
                archiveArtifacts artifacts: 'artifacts/train.csv, artifacts/test.csv, artifacts/eval.csv', fingerprint: true
            }
        }

        stage('Run Train') {
            steps {
                script {
                    sh "uv run python src/train.py"

                    def runId = sh(
                        script: "cat artifacts/run_id.txt",
                        returnStdout: true
                    ).trim()

                    sh "uv run python src/predict.py --model-uri runs:/${runId}/model"
                }
            }
        }

        stage('Archive Model And Predictions') {
            steps {
                archiveArtifacts artifacts: 'artifacts/savePred.txt,artifacts/run_id.txt,mlruns/**', fingerprint: true
            }
        }
    }
}