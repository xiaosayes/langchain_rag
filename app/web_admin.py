"""Interactive bootstrap for the single web administrator."""

import getpass

from app.bootstrap import state_directory
from app.config import load_settings
from app.db.session import create_session_factory
from app.web.auth import AuthService, DuplicateAdminError


def main() -> None:
    """Prompt for credentials; passwords are never command-line arguments."""
    username = input("Username: ").strip()
    password = getpass.getpass("Password: ")
    confirmation = getpass.getpass("Confirm password: ")
    if password != confirmation:
        raise SystemExit("Passwords do not match")

    settings = load_settings()
    root = state_directory()
    root.mkdir(parents=True, exist_ok=True)
    service = AuthService(
        create_session_factory(root / "index.sqlite3"),
        session_ttl_hours=settings.web.session_ttl_hours,
    )
    try:
        service.create_admin(username, password)
    except (DuplicateAdminError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"Administrator {username!r} created.")


if __name__ == "__main__":
    main()
