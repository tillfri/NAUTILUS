import os

from biigle import Api

api = Api(email="your.name@ostfalia.de", token="token")


def download_video(video_id, output_filename):
    """Download a video file from BIIGLE.

    Args:
        video_id: The BIIGLE video ID.
        output_filename: Path where the video file should be saved.

    Raises:
        requests.HTTPError: If the download fails.
    """
    url = "videos/{}/file".format(video_id)
    response = api.get(url, stream=True)

    with open(output_filename, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)


def get_project_video_count(project_id):
    """Get video counts for all volumes in a project.

    Args:
        project_id: The BIIGLE project ID.

    Returns:
        List of video lists for each volume.
    """
    volumes = api.get(f"projects/{project_id}/volumes").json()
    video_counter = []
    for vol in volumes:
        videos = api.get("volumes/{}/files".format(vol["id"])).json()
        video_counter.append(videos)
    return video_counter


def find_missing_video_dirs(project_id, base_dir):
    """Find video IDs that exist in BIIGLE but not as local directories.

    Args:
        project_id: The BIIGLE project ID.
        base_dir: Local directory path to check for video subdirectories.

    Returns:
        Set of missing video IDs.
    """
    volumes = api.get(f"projects/{project_id}/volumes").json()

    api_video_ids = set()
    for vol in volumes:
        videos = api.get(f"volumes/{vol['id']}/files").json()
        api_video_ids.update(str(v) for v in videos)

    local_video_ids = {name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))}

    missing_video_ids = api_video_ids - local_video_ids
    return missing_video_ids


def get_video_annotations(project_id):
    """Retrieve annotations for all videos in a project.

    Args:
        project_id: The BIIGLE project ID.

    Returns:
        Dict mapping video IDs to their annotation data.
    """
    volumes = api.get(f"projects/{project_id}/volumes").json()
    annotations_by_video = {}

    for vol in volumes:
        videos = api.get("volumes/{}/files".format(vol["id"])).json()
        for video in videos:
            url = "videos/{}/annotations".format(video)
            response = api.get(url)
            if response.status_code == 200:
                annotations_by_video[video] = response.json()

    return annotations_by_video


if __name__ == "__main__":
    missing = find_missing_video_dirs("3764", "images_sorted_by_vid_id_AFTER_AB")
    if missing:
        print(f"Found {len(missing)} missing video directories:")
        for video_id in sorted(missing):
            print(f"  {video_id}")
