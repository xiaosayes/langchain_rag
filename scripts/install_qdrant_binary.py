"""Install the official Qdrant 1.15.4 standalone binary without Docker."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

VERSION = "1.15.4"
RELEASE_API = f"https://api.github.com/repos/qdrant/qdrant/releases/tags/v{VERSION}"
USER_AGENT = "enterprise-rag-qdrant-installer"
ASSETS = {
    "windows": (
        "qdrant-x86_64-pc-windows-msvc.zip",
        "application/zip",
        "qdrant.exe",
    ),
    "ubuntu": (
        "qdrant-x86_64-unknown-linux-gnu.tar.gz",
        "application/x-gtar",
        "qdrant",
    ),
}


def _host_asset() -> tuple[str, str, str]:
    machine = platform.machine().lower()
    if machine not in {"amd64", "x86_64"}:
        raise RuntimeError(f"Unsupported CPU architecture: {platform.machine()}")
    if sys.platform == "win32":
        return ASSETS["windows"]
    if sys.platform.startswith("linux"):
        try:
            release = platform.freedesktop_os_release()
        except OSError as error:
            raise RuntimeError("Cannot identify this Linux distribution") from error
        if release.get("ID") != "ubuntu" or release.get("VERSION_ID") != "22.04":
            raise RuntimeError(
                "The standalone Linux installer supports only Ubuntu 22.04 x86_64; "
                f"detected {release.get('PRETTY_NAME', 'unknown Linux')}"
            )
        return ASSETS["ubuntu"]
    raise RuntimeError(f"Unsupported development host: {sys.platform}")


def _open(url: str):
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": USER_AGENT,
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    last_error: urllib.error.URLError | None = None
    for attempt in range(3):
        try:
            return urllib.request.urlopen(request, timeout=60)
        except urllib.error.URLError as error:
            last_error = error
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError(f"Could not download {url}: {last_error}")


def _release_asset(asset_name: str, expected_type: str) -> dict:
    try:
        with _open(RELEASE_API) as response:
            release = json.load(response)
    except RuntimeError as urllib_error:
        curl = shutil.which("curl")
        if curl is None:
            raise
        result = subprocess.run(
            [
                curl,
                "--fail",
                "--location",
                "--retry",
                "3",
                "--connect-timeout",
                "30",
                "--max-time",
                "120",
                "--header",
                f"User-Agent: {USER_AGENT}",
                RELEASE_API,
            ],
            check=False,
            capture_output=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Python GitHub API request failed ({urllib_error}); "
                f"curl exited {result.returncode}"
            ) from urllib_error
        release = json.loads(result.stdout)
    if release.get("tag_name") != f"v{VERSION}":
        raise RuntimeError("GitHub returned the wrong Qdrant release")
    matches = [asset for asset in release["assets"] if asset["name"] == asset_name]
    if len(matches) != 1:
        raise RuntimeError(f"Official release asset is missing: {asset_name}")
    asset = matches[0]
    if asset.get("content_type") != expected_type:
        raise RuntimeError(
            f"Unexpected type for {asset_name}: {asset.get('content_type')!r}"
        )
    if type(asset.get("size")) is not int or asset["size"] < 1024 * 1024:
        raise RuntimeError(f"Unexpected size for {asset_name}: {asset.get('size')!r}")
    if not str(asset.get("browser_download_url", "")).startswith(
        f"https://github.com/qdrant/qdrant/releases/download/v{VERSION}/"
    ):
        raise RuntimeError("Release asset does not use the official Qdrant download URL")
    return asset


def _download(asset: dict, destination: Path) -> str:
    url = asset["browser_download_url"]
    try:
        with _open(url) as response, destination.open("xb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
    except RuntimeError as urllib_error:
        # Some Windows development networks interrupt Python's TLS stack while
        # the system curl client works. The same official URL and checks apply.
        curl = shutil.which("curl")
        if curl is None:
            raise
        destination.unlink(missing_ok=True)
        result = subprocess.run(
            [
                curl,
                "--fail",
                "--location",
                "--retry",
                "3",
                "--connect-timeout",
                "30",
                "--max-time",
                "300",
                "--output",
                str(destination),
                url,
            ],
            check=False,
        )
        if result.returncode != 0:
            destination.unlink(missing_ok=True)
            raise RuntimeError(
                f"Python download failed ({urllib_error}); "
                f"curl exited {result.returncode}"
            ) from urllib_error
    size = destination.stat().st_size
    if size != asset["size"]:
        raise RuntimeError(
            f"Downloaded {size} bytes; GitHub metadata declares {asset['size']}"
        )
    digest = hashlib.sha256()
    with destination.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _check_archive_type(archive: Path, executable_name: str) -> None:
    with archive.open("rb") as source:
        signature = source.read(4)
    if executable_name == "qdrant.exe":
        if not signature.startswith(b"PK\x03\x04"):
            raise RuntimeError("Windows release asset is not a ZIP archive")
    elif not signature.startswith(b"\x1f\x8b"):
        raise RuntimeError("Ubuntu release asset is not a gzip-compressed tar archive")


def _safe_member_path(root: Path, name: str) -> Path:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError(f"Unsafe archive member path: {name}")
    destination = (root / Path(*path.parts)).resolve()
    if root.resolve() not in destination.parents and destination != root.resolve():
        raise RuntimeError(f"Archive member escapes extraction root: {name}")
    return destination


def _extract(archive: Path, root: Path, executable_name: str) -> None:
    if executable_name == "qdrant.exe":
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                _safe_member_path(root, member.filename)
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode):
                    raise RuntimeError("Symbolic links are not accepted in the archive")
            bundle.extractall(root)
        return
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            _safe_member_path(root, member.name)
            if member.issym() or member.islnk() or member.isdev():
                raise RuntimeError("Links and devices are not accepted in the archive")
        bundle.extractall(root, filter="data")


def _executable(root: Path, name: str) -> Path:
    matches = [path for path in root.rglob(name) if path.is_file()]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one {name} in the official archive")
    executable = matches[0]
    if name == "qdrant":
        executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    return executable


def _check_version(executable: Path) -> str:
    try:
        result = subprocess.run(
            [str(executable), "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except OSError as error:
        raise RuntimeError(f"Could not execute {executable}: {error}") from error
    output = f"{result.stdout}\n{result.stderr}".strip()
    if result.returncode != 0 or not re.search(
        rf"(?<![\d.]){re.escape(VERSION)}(?![\d.])", output
    ):
        raise RuntimeError(
            f"Qdrant version check failed (exit {result.returncode}): {output}"
        )
    return output


def install(root: Path) -> Path:
    asset_name, expected_type, executable_name = _host_asset()
    destination = root / VERSION
    executable = destination / executable_name
    if destination.exists():
        if not executable.is_file():
            raise RuntimeError(f"Incomplete existing installation: {destination}")
        output = _check_version(executable)
        print(f"Existing installation verified: {output}")
        print(executable)
        return executable

    root.mkdir(parents=True, exist_ok=True)
    asset = _release_asset(asset_name, expected_type)
    with tempfile.TemporaryDirectory(prefix=".qdrant-install-", dir=root) as temporary:
        staging = Path(temporary)
        archive = staging / asset_name
        archive_sha256 = _download(asset, archive)
        _check_archive_type(archive, executable_name)
        package = staging / "package"
        package.mkdir()
        _extract(archive, package, executable_name)
        staged_executable = _executable(package, executable_name)
        version_output = _check_version(staged_executable)
        shutil.copy2(archive, package / asset_name)
        manifest = {
            "version": VERSION,
            "tag": f"v{VERSION}",
            "release_api": RELEASE_API,
            "asset_name": asset_name,
            "asset_url": asset["browser_download_url"],
            "asset_content_type": asset["content_type"],
            "asset_size": asset["size"],
            "archive_sha256": archive_sha256,
            "executable": executable_name,
            "version_output": version_output,
            "installed_at": datetime.now(UTC).isoformat(),
        }
        (package / "install-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        shutil.move(str(package), str(destination))

    print(f"Installed official Qdrant {VERSION}: {executable}")
    print(f"Archive SHA-256: {archive_sha256}")
    print(f"Version check: {version_output}")
    return executable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("var/qdrant"),
        help="Installation root (default: var/qdrant)",
    )
    arguments = parser.parse_args()
    try:
        install(arguments.root.resolve())
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile, tarfile.TarError) as error:
        parser.exit(1, f"error: {error}\n")


if __name__ == "__main__":
    main()
