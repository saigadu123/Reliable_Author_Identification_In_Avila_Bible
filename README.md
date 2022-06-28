# Reliable_Author_Identification_In_Avila_Bible
A Machine Learning Project where we predict the Author 


Create Conda Environment
```
conda create -p venv python==3.7
```
Activate Conda Environment

```
conda activate venv/

```
Create requirements.txt file
```
pip install -r requirements.txt

```
Git commands
```
git add <filename> --> To add files to git
```
git add . ---> To add all files into git
```
git status --> TO check all version maintained by Git
```
.gitignore file --> To ignore file or folder we add /<filename> in gitignore file
```
git log --> To check the versions
```
git commit -m "message" --> TO create version/commit all changes by git
```
git push origin main --> To send version/changes into github
```
git remote -v --> To check remote url
```
TO setup CI/CD pipeline in heroku we need 3 information

HEROKU_EMAIL = saikrishnagorantla2001@gmail.com
HEROKU_API_KEY = eada18ed-9702-4533-917d-49a9edd2b8e5
HEROKU_APP_NAME = ml-regression-sai

BUILD DOCKER IMAGE

```
docker build -t <image_name>:<tagname> .
```
Note: imagename must be in lowercase

To list docker image
```
docker images
```
Run docker image
```
docker run -p 5000:5000 -e PORT=5000 f8c749e73678
```
To check running container
```
docker ps
```
To stop docker container
```
docker stop <container-id>
```
```
Run python setup.py install after creating setup.py file
```
Install ipykernel
```
pip install ipykernel

```