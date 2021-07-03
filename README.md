# got_extractiveqa
Extractive question & answering based on Game of Thrones wikipedia articles

# Installation

`docker compose up`

# Usage

After running the container for the first time, please run the following command to update the data:
`http://localhost:8777/update_docustore`

Afterwards, you can ask any GoT question like so:

`http://localhost:8777/qna/"who is the father of Arya Stark"?`
