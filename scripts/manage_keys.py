#!/usr/bin/env python3

import argparse
import sys
from datetime import datetime, timedelta
import os
from typing import List, Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.auth import APIKeyManager, APIKey

def create_key(owner: str, expires_in_days: int, permissions: List[str]) -> APIKey:
    """Create a new API key."""
    manager = APIKeyManager()
    key = manager.generate_key(
        owner=owner,
        expires_at=datetime.utcnow() + timedelta(days=expires_in_days),
        permissions=permissions
    )
    return key

def list_keys(owner: Optional[str] = None, show_expired: bool = False) -> List[APIKey]:
    """List all API keys, optionally filtered by owner."""
    manager = APIKeyManager()
    keys = manager.list_keys(owner=owner)
    if not show_expired:
        keys = [k for k in keys if k.is_active and k.expires_at > datetime.utcnow()]
    return keys

def revoke_key(key_id: str) -> bool:
    """Revoke an API key."""
    manager = APIKeyManager()
    return manager.revoke_key(key_id)

def main():
    parser = argparse.ArgumentParser(description='Manage MCP API keys')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Create key command
    create_parser = subparsers.add_parser('create', help='Create a new API key')
    create_parser.add_argument('--owner', required=True, help='Owner email')
    create_parser.add_argument('--expires', type=int, default=90, help='Expiration in days')
    create_parser.add_argument('--permissions', nargs='+', default=['read', 'write'],
                             help='List of permissions')

    # List keys command
    list_parser = subparsers.add_parser('list', help='List API keys')
    list_parser.add_argument('--owner', help='Filter by owner')
    list_parser.add_argument('--show-expired', action='store_true',
                            help='Include expired keys')

    # Revoke key command
    revoke_parser = subparsers.add_parser('revoke', help='Revoke an API key')
    revoke_parser.add_argument('key_id', help='Key ID to revoke')

    args = parser.parse_args()

    if args.command == 'create':
        key = create_key(args.owner, args.expires, args.permissions)
        print(f"\nCreated new API key:")
        print(f"Key ID: {key.key_id}")
        print(f"API Key: {key.key}")
        print(f"Owner: {key.owner}")
        print(f"Expires: {key.expires_at}")
        print(f"Permissions: {', '.join(key.permissions)}")
        print("\nIMPORTANT: Store this API key securely. It won't be shown again.")

    elif args.command == 'list':
        keys = list_keys(args.owner, args.show_expired)
        print(f"\nFound {len(keys)} API keys:")
        for key in keys:
            status = "ACTIVE" if key.is_active else "REVOKED"
            if key.expires_at < datetime.utcnow():
                status = "EXPIRED"
            print(f"\nKey ID: {key.key_id}")
            print(f"Owner: {key.owner}")
            print(f"Status: {status}")
            print(f"Created: {key.created_at}")
            print(f"Expires: {key.expires_at}")
            print(f"Last Used: {key.last_used or 'Never'}")
            print(f"Usage Count: {key.usage_count}")
            print(f"Permissions: {', '.join(key.permissions)}")

    elif args.command == 'revoke':
        if revoke_key(args.key_id):
            print(f"Successfully revoked key: {args.key_id}")
        else:
            print(f"Failed to revoke key: {args.key_id}")
            sys.exit(1)

    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main() 