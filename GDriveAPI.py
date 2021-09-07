from google.cloud import storage
# from bookshelf import get_model, storage


from flask import Blueprint, current_app, redirect, render_template, request, url_for

def explicit():

    storage_client = storage.Client.from_service_account_json('firstProject.json')

    # Make an authenticated API request
    buckets = list(storage_client.list_buckets())
    bucket = storage_client.get_bucket('staging.optical-psyche-233422.appspot.com')

    blob = bucket.blob('saverstal.py')
    blob.upload_from_string('this is test content')

    print(buckets)

    buckets.append('checking.py')
    print('appended')
    print(buckets)


def upload_file(file):
    """
    Upload the user-uploaded file to Google Cloud Storage and retrieve its
    publicly-accessible URL.
    """
    if not file:
        return None

    public_url = storage.upload_file(
        file.read(),
        file.filename,
        file.content_type
    )

    current_app.logger.info(
        "Uploaded file %s as %s.", file.filename, public_url)

    return public_url

file_url = upload_file(request.files.get('saverstal 4.py'))



# explicit()