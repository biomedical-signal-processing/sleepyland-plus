"""Validate uploaded EDF file selections before they reach the inference API."""

def validate_files(files):
    """
    Validate a list of uploaded files from a request.

    This helper performs a very basic check:
      - Ensures that `files` is not empty or None.
      - Ensures that none of the files has an empty filename.

    It is designed to be used early in a route to guard against invalid uploads
    before doing any further processing (saving files, launching predictions, etc.).

    Parameters
    ----------
    files : list of Werkzeug FileStorage or None
        The list of files obtained from something like:
          request.files.getlist('edf-files')

    Returns
    -------
    tuple
        (is_valid, error_response, status_code) where:
          - is_valid : bool
                True if validation passes, False otherwise.
          - error_response : dict or None
                If invalid, a dictionary describing the error
                (e.g., {'error': 'No files provided or empty filenames'}).
                If valid, this is None.
          - status_code : int or None
                HTTP status code to return in case of error (e.g., 400 for bad request).
                If valid, this is None.

    Example
    -------
    is_valid, error_body, status = validate_files(uploaded_files)
    if not is_valid:
        return jsonify(error_body), status
    """
    # Check if the list is empty/None OR if any file has an empty filename
    if not files or any(file.filename == '' for file in files):
        # Invalid: return False, an error message, and HTTP 400 (Bad Request)
        return False, {'error': 'No files provided or empty filenames'}, 400

    # Valid: no error message or status code needed
    return True, None, None
