"""Manage input and output directories shared by the Docker services."""

import os      # For filesystem path operations and directory listing
import shutil  # For removing directories recursively


def save_uploaded_files(files):
    """
    Save uploaded files to /app/input, preserving their relative paths.

    This function appears to be a generic uploader where each incoming file may
    have a 'relative path' encoded in file.filename (e.g. "folder/file.edf").
    It uses that path to recreate the same directory structure under /app/input.

    Parameters
    ----------
    files : list of Werkzeug FileStorage
        List of file objects from Flask's request.files (e.g. request.files.getlist(...)).

    Returns
    -------
    str
        The 'folder_root_name', i.e., the first path component of the last processed file's
        relative path. For example, if file.filename is "myedf/file1.edf", this returns "myedf".
        This can be used later to identify the top-level folder where the dataset was stored.
    """
    folder_root_name = ""
    for file in files:
        # The filename may include a subdirectory path (e.g., "myedf/file1.edf")
        relative_path = file.filename

        # The "root" folder name is the first part before '/' (e.g., "myedf")
        folder_root_name = relative_path.split("/")[0]

        # Build the full path where the file will be stored under /app/input
        save_path = os.path.join('/app/input', relative_path)

        # Ensure that the directory structure exists before saving the file
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save the uploaded file data to disk
        file.save(save_path)

    # Return the last (or only) root folder name encountered
    return folder_root_name


def save_edf_files(files, folder_name, log):
    """
    Save EDF files into /app/input/<folder_name>, with logging.

    This function uses the user-specified folder_name so that each run
    can have its own dedicated input subfolder.

    This is called from the /process route after reading folderName
    from the form.

    Parameters
    ----------
    files : list of Werkzeug FileStorage
        EDF files posted from the GUI (e.g. request.files.getlist('edf-files')).

    folder_name : str
        Name chosen by the user in the GUI (e.g. "patient_batch_01"). All EDFs
        for this run will be saved under /app/input/<folder_name>.

    log : logging.Logger
        Logger for debug/info/error messages about the file saving process.
    """
    # Target directory where all EDF files for this run will be saved.
    # This path is shared with other containers via Docker volumes.
    # Example: /app/input/patient_batch_01
    save_dir = f'/app/input/{folder_name}'

    # Ensure that /app/input/<folder_name> exists (create it if it doesn't)
    os.makedirs(save_dir, exist_ok=True)

    # Loop over each uploaded file
    for file in files:
        log.debug(f"Processing file: {file.filename}")

        # Build the full path for the file (e.g., /app/input/patient_batch_01/subject1.edf)
        file_path = os.path.join(save_dir, file.filename)
        log.debug(f"Saving to: {file_path}")

        try:
            # Save the file content to disk
            file.save(file_path)
            log.info(f"File saved: {file_path}")
        except Exception as e:
            # Log any failure to save an uploaded file
            log.error(f"Error saving file {file.filename}: {e}")


def clear_directory(directory: str):
    """
    Delete all files and subdirectories inside the given directory.

    This is used by:
      - /reset-directories endpoint (and indirectly by /process) to clear ./input
        before a new run, ensuring no leftover files interfere with the next prediction.

    Parameters
    ----------
    directory : str
        Path to the directory whose contents should be removed.
    """
    # Iterate over all entries inside the directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            # If entry is a regular file or a symbolic link, delete it
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            # If entry is a subdirectory, delete the entire directory tree
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            # On failure, print an error message (could also be logged instead)
            print(f"Failed to delete {file_path}. Reason: {e}")
