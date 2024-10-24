#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Python file for queriying StringDB (https://string-db.org/)."""

import datetime
import logging
import warnings

import numpy as np
import pandas as pd
import requests
import time

from pyBiodatafuse.constants import (
    NCBI_ENDPOINT,
    STRING,
    STRING_ENDPOINT,
    STRING_GENE_INPUT_ID,
    STRING_OUTPUT_DICT,
    STRING_PPI_COL,
    UNIPROT_ENDPOINT
)
from pyBiodatafuse.utils import check_columns_against_constants, get_identifier_of_interest

logger = logging.getLogger("stringdb")


def check_endpoint_stringdb() -> bool:
    """Check the availability of the STRING Db endpoint.

    :returns: True if the endpoint is available, False otherwise.
    """
    response = requests.get(f"{STRING_ENDPOINT}/json/version")

    # Check if API is down
    if response.status_code == 200:
        return True
    else:
        return False


def get_version_stringdb() -> dict:
    """Get version of STRING-DB API.

    :returns: a dictionary containing the version information
    """
    version_call = requests.get(f"{STRING_ENDPOINT}/json/version").json()
    return {"source_version": version_call[0]["string_version"]}


def _format_data(row, string_ids_df, network_df, to_uniprot):
    """Reformat STRING-DB response (Helper function).

    :param string_ids_df: dataframe containing string identifiers
    :param row: input_df row
    :param network_df: STRING-DB response annotation DataFrame
    :param network_df: list of uniprot gene identifiers
    :returns: StringDB reformatted annotation.
    """

    gene_ppi_links = list()

    target_links_set = set()


    for _i, row_str in string_ids_df.iterrows():
        for _i, row_arr in network_df.iterrows():

            if row_arr["preferredName_A"] == row_str["preferredName"] and row["identifier"] == row_str["queryItem"]:
                if row_arr["preferredName_B"] not in target_links_set:
                    gene_ppi_links.append(
                        {
                            "stringdb_link_to": row_arr["preferredName_B"],
                            STRING_GENE_INPUT_ID: row_arr["stringId_B"].split(".")[1],
                            "score": row_arr["score"],
                            "string_id": row_arr["stringId_B"]
                        }
                    )
                    target_links_set.add(row_arr["preferredName_B"])
                    to_uniprot.append(row_arr["stringId_B"])

            elif row_arr["preferredName_B"] == row_str["preferredName"] and row["identifier"] == row_str["queryItem"]:
                if row_arr["preferredName_A"] not in target_links_set:
                    gene_ppi_links.append(
                        {
                            "stringdb_link_to": row_arr["preferredName_A"],
                            STRING_GENE_INPUT_ID: row_arr["stringId_A"].split(".")[1],
                            "score": row_arr["score"],
                            "string_id": row_arr["stringId_A"]
                        }
                    )
                    target_links_set.add(row_arr["preferredName_A"])
                    to_uniprot.append(row_arr["stringId_A"])

    return gene_ppi_links


def get_string_ids(gene_list: list, species: int = 9606) -> str:
    """Get the String identifiers of the gene list.
    :param gene_list: list containing the genes of interest.
    :param species: input species NCBI Taxonomy ID
    :returns results: list where each item is a dictionary containing information of a gene of interest
    """
    params = {
        "identifiers": "\r".join(gene_list),  # your protein list
        "species": species,  #  species NCBI identifier (default: human)
        "limit": 1,  # only one (best) identifier per input protein
        "caller_identity": "github.com",  # your app name
        "echo_query" : 1, # column with your input identifier
    }

    results = requests.post(f"{STRING_ENDPOINT}/json/get_string_ids", data=params).json()
    return results


def _get_ppi_data(gene_ids: list, species: int = 9606) -> pd.DataFrame:
    """Get the STRING PPI interactions for the gene list for a specific species.
    :param gene_ids: list of STRING identifiers.
    :param species: input species NCBI Taxonomy ID
    :returns response: list where each item is a dictionary containing PPI data. 
    """
    params = {
        "identifiers": "%0d".join(gene_ids),  # your protein
        "species": species,  # species NCBI identifier (default: human)
        "caller_identity": "github.com",  # your app name
    }
    
    response = requests.post(f"{STRING_ENDPOINT}/json/network", data=params).json()

    return response


def get_uniprot_ids(string_ids):
    """Get the UniProt IDs using a list of STRING IDs.
    :param string_ids: list containing STRING identifiers
    :returns mapped_ids: dictionary with the string ids as key, and the uniprot ids as value
    """
    # Submit ID mapping request
    request = requests.post(
        f"{UNIPROT_ENDPOINT}/idmapping/run",
        data={"from": "STRING", "to": "UniProtKB", "ids": ",".join(string_ids)},
    )

    if request.status_code != 200:
        raise Exception(f"Error: {request.status_code}, {request.text}")

    job_id = request.json().get("jobId")

    # While loop to check if the job is finished
    while True:
        status_request = requests.get(f"{UNIPROT_ENDPOINT}/idmapping/status/{job_id}")
        status = status_request.json()
        if status.get("results") or status.get("failedIds"):
            break
        time.sleep(5)  # Wait for 5 seconds before checking again

    # When the job is finished, get the results
    results_request = requests.get(f"{UNIPROT_ENDPOINT}/idmapping/results/{job_id}")

    results = results_request.json()

    # Check for results and map STRING to UniProt
    if "results" not in results:
        raise Exception(f"No results found in the mapping response: {results}")

    mapped_ids = {result["from"]: result["to"] for result in results["results"]}

    return mapped_ids


def get_ppi(bridgedb_df: pd.DataFrame, species: str = "human"):
    """Annotate genes with protein-protein interactions from STRING-DB.

    :param bridgedb_df: BridgeDb output for creating the list of gene ids to query
    :param species: The species to query. All species that are supported by both NCBI and STRINGDB can be used.
    :returns: a DataFrame containing the StringDB output and dictionary of the metadata.
    """
    # Retrieve NCBI taxonomy identifier
    params = {
        "db": "taxonomy",
        "term": species,
        "retmode": "json" 
    }

    response = requests.get(f"{NCBI_ENDPOINT}/entrez/eutils/esearch.fcgi", params=params).json()
    species_id = response["esearchresult"]["idlist"][0]

    # Check if the endpoint is available
    api_available = check_endpoint_stringdb()
    if not api_available:
        warnings.warn(f"{STRING} endpoint is not available. Unable to retrieve data.", stacklevel=2)
        return pd.DataFrame(), {}

    string_version = get_version_stringdb()

    # Record the start time
    start_time = datetime.datetime.now()

    data_df = get_identifier_of_interest(bridgedb_df, STRING_GENE_INPUT_ID)
    data_df = data_df.reset_index(drop=True)

    gene_list = list(set(data_df["target"].tolist()))

    # Return empty dataframe when only one input submitted
    if len(gene_list) == 1:
        warnings.warn(
            f"There is only one input gene/protein. Provide at least two input to extract their interactions from {STRING}.",
            stacklevel=2,
        )
        return pd.DataFrame(), {}

    # Get STRING IDs
    string_ids = get_string_ids(gene_list, species_id)
    stringdb_ids_df = pd.DataFrame(string_ids)
    stringdb_ids_df.queryIndex = stringdb_ids_df.queryIndex.astype(str)

    # Get the PPI data
    response = _get_ppi_data(list(stringdb_ids_df.stringId.unique()), species_id)
    network_df = pd.DataFrame(response)

    # Record the end time
    end_time = datetime.datetime.now()

    """Metadata details"""
    # Get the current date and time
    current_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Calculate the time elapsed
    time_elapsed = str(end_time - start_time)
    # Calculate the number of new edges
    num_new_edges = network_df.drop_duplicates(subset=["stringId_A", "stringId_B"]).shape[0]

    # Check the network_df
    if num_new_edges != len(network_df):
        warnings.warn(
            f"The network_df in {STRING} annotator should be checked. Please create an issue https://github.com/BioDataFuse/pyBiodatafuse/issues/.",
            stacklevel=2,
        )

    # Add the datasource, query, query time, and the date to metadata
    string_metadata = {
        "datasource": STRING,
        "metadata": {"source_version": string_version},
        "query": {
            "size": len(gene_list),
            "input_type": STRING_GENE_INPUT_ID,
            "number_of_added_edges": num_new_edges,
            "time": time_elapsed,
            "date": current_date,
            "url": STRING_ENDPOINT,
        },
    }

    if "stringId_A" not in network_df.columns:
        warnings.warn(
            f"There is no interaction between your input list based on {STRING}, {string_version}.",
            stacklevel=2,
        )
        return pd.DataFrame(), string_metadata

    # Format the data
    to_uniprot = list()
    data_df[STRING_PPI_COL] = data_df.apply(lambda row: _format_data(row, stringdb_ids_df, network_df, to_uniprot), axis=1)

    # Get UniProt identifiers
    uniprot_ids = get_uniprot_ids(to_uniprot)

    # Append the uniprot identifiers to the current dataframe
    data_df[STRING_PPI_COL] = data_df[STRING_PPI_COL].apply(
    lambda ppi_list: [
        {**ppi, 'uniprot_id': uniprot_ids.get(ppi.get('string_id'), None)} for ppi in ppi_list
    ]
    )

    data_df[STRING_PPI_COL] = data_df[STRING_PPI_COL].apply(
        lambda x: ([{key: np.nan for key in STRING_OUTPUT_DICT.keys()}] if len(x) == 0 else x)
    )

    # Check if all keys in df match the keys in OUTPUT_DICT
    exploded_df = data_df.explode(STRING_PPI_COL)
    ppi_df = pd.json_normalize(exploded_df[STRING_PPI_COL])
    check_columns_against_constants(
        data_df=ppi_df,
        output_dict=STRING_OUTPUT_DICT,
        check_values_in=[],
    )

    return data_df, string_metadata
