"""Delete the Vector Search resources that setup_vector_search.py created.

A deployed index is billed for every hour it stays deployed, whether or not
you search it. This script undeploys and deletes the index endpoint, then
deletes the index. Without --yes it only lists what it would delete.

The bucket and the Document AI processors are not touched; the README's
"Clean up" section shows how to delete them.
"""

import argparse
import os
import sys


def find_resources(index_name):
    from google.cloud import aiplatform
    endpoints = aiplatform.MatchingEngineIndexEndpoint.list(
        filter=f'display_name="{index_name}-endpoint"')
    indexes = aiplatform.MatchingEngineIndex.list(filter=f'display_name="{index_name}"')
    return endpoints, indexes


def main(project_id, location, index_name, yes=False):
    from google.cloud import aiplatform
    aiplatform.init(project=project_id, location=location)

    endpoints, indexes = find_resources(index_name)
    if not endpoints and not indexes:
        print(f"No index or endpoint named '{index_name}' in {project_id}/{location}. Nothing to delete.")
        return
    for e in endpoints:
        deployed = [d.id for d in e.deployed_indexes]
        print(f"Endpoint: {e.resource_name} (deployed indexes: {', '.join(deployed) or 'none'})")
    for i in indexes:
        print(f"Index:    {i.resource_name}")

    if not yes:
        print("\nDry run. Add --yes to undeploy and delete these resources.")
        return

    for e in endpoints:
        print(f"Undeploying all indexes from {e.display_name} (this can take a few minutes)...")
        e.undeploy_all()
        e.delete()
        print(f"Deleted endpoint {e.resource_name}")
    for i in indexes:
        i.delete()
        print(f"Deleted index {i.resource_name}")
    print("Vector Search teardown complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Undeploy and delete the Vector Search index and endpoint")
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT", "YOUR_PROJECT_ID"), help="GCP project ID")
    parser.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"), help="GCP region")
    parser.add_argument("--index-name", default="kyc-documents-index", help="Vector search index name")
    parser.add_argument("--yes", action="store_true", help="Delete (without it, only list)")
    args = parser.parse_args()
    try:
        main(args.project, args.location, args.index_name, args.yes)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
