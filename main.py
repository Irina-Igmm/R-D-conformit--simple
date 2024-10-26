from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import os

 # Import the class
from document_classification import ClassificationDocument 
from extract_info import ExtractInfo

app = FastAPI()
API_KEY = os.getenv("OPENAI_API_KEY")


@app.post("/classify-document/")
async def classify_document(file: UploadFile = File(...)):
    """Endpoint for classifying a PDF or image."""
    if not (
        file.filename.endswith(".pdf")
        or file.filename.endswith((".png", ".jpg", ".jpeg"))
    ):
        raise HTTPException(
            status_code=400, detail="Please provide a valid PDF or image."
        )

    temp_file_path = f"./{file.filename}"
    try:
        # Save the file temporarily
        with open(temp_file_path, "wb") as f:
            f.write(await file.read())

        # Initialize the classifier with the JSON file
        # Initialisation
        classifier = ClassificationDocument(
           json_file_path="./Tools/list_type_doc.json",
           location=os.getenv("LOCATION"),
           service_account_path=os.getenv("VERTEX_AI_KEY"),
        )

        extract_info = ExtractInfo(
            temp_file_path=temp_file_path,)

        # Extract text based on file type
        if file.filename.endswith(".pdf"):
            document_content = extract_info.extract_text_from_pdf(temp_file_path)
        else:
            document_content = extract_info.extract_text_from_image(temp_file_path)

        # Classify the document and return the result
        result = classifier.classify(document_content)
        return JSONResponse(content={"result": result})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Remove the temporary file after processing
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# Entry point for running the FastAPI application with Uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
