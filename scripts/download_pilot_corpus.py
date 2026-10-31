#!/usr/bin/env python3
"""Download the 21 pilot corpus PDFs into the repository's data directory."""

import argparse
from pathlib import Path
import shutil
import sys
import tempfile
from urllib.request import Request, urlopen


REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEOUT_SECONDS = 60
# World Bank's Cloudflare frontend rejects urllib's default User-Agent (1010).
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept": "application/pdf"}
# (split, country_slug, filename, pdf_url)
MANIFEST = (
    ('input', 'mexico', 'MX_PAD_2011.pdf',
     'https://documents.worldbank.org/curated/en/478321468057257305/pdf/637340PAD0P1230Official0Use0Only090.pdf'),
    ('input', 'mexico', 'MX_ISR05_2014.pdf',
     'https://documents.worldbank.org/curated/en/167041468270295147/pdf/ISR-Disclosable-P123367-07-04-2014-1404530311278.pdf'),
    ('input', 'mexico', 'MX_ISR08_2016.pdf',
     'https://documents.worldbank.org/curated/en/582041468055778739/pdf/ISR-Disclosable-P123367-04-04-2016-1459774743959.pdf'),
    ('input', 'mexico', 'MX_ISR11_2017.pdf',
     'https://documents.worldbank.org/curated/en/798871500576648488/pdf/ISR-Disclosable-P123367-07-20-2017-1500576633287.pdf'),
    ('holdout', 'mexico', 'MX_ICR_HOLDOUT_2018.pdf',
     'https://documents.worldbank.org/curated/en/581671516112715944/pdf/ICR00004310-01122018.pdf'),
    ('holdout', 'mexico', 'MX_IEG_HOLDOUT_2018.pdf',
     'https://documents.worldbank.org/curated/en/710091526397044238/pdf/Mexico-MX-Savings-and-Credit-Sector-Loan.pdf'),
    ('input', 'comoros', 'KM_PAD_2020.pdf',
     'https://documents.worldbank.org/curated/en/662311592667181272/pdf/Comoros-Financial-Inclusion-Project.pdf'),
    ('input', 'comoros', 'KM_ISR03_2021.pdf',
     'https://documents.worldbank.org/curated/en/929091636652478577/pdf/Disclosable-Version-of-the-ISR-Comoros-Financial-Inclusion-Project-P166193-Sequence-No-03.pdf'),
    ('input', 'comoros', 'KM_ISR07_2023.pdf',
     'https://documents.worldbank.org/curated/en/099091123063538146/pdf/P1661930b71e4a032086c90e0b32b67fe50.pdf'),
    ('input', 'comoros', 'KM_RESTRUCTURING_2024.pdf',
     'https://documents.worldbank.org/curated/en/099061924121020448/pdf/P166193-1f8bc082-93f2-428c-8150-d9bd0538e193.pdf'),
    ('input', 'comoros', 'KM_ISR10_2025.pdf',
     'https://documents.worldbank.org/curated/en/099033125133030976/pdf/P166193-c1f6fb2c-e0a1-4329-bac2-c53bef500942.pdf'),
    ('holdout', 'comoros', 'KM_ICR_HOLDOUT_2026.pdf',
     'https://documents.worldbank.org/curated/en/099032326091520607/pdf/BOSIB-cd362a8b-2323-4f0d-8e18-4073ca0c5ea8.pdf'),
    ('holdout', 'comoros', 'KM_IEG_HOLDOUT_2026.pdf',
     'https://documents.worldbank.org/curated/en/099092426124042623/pdf/P166193-b97a52ae-f6d1-4879-8d15-3c105c26d47b.pdf'),
    ('input', 'sierra-leone', 'SL_PAD_2014.pdf',
     'https://documents.worldbank.org/curated/en/211781468101070719/pdf/825790PAD0P143010Box382156B00OUO090.pdf'),
    ('input', 'sierra-leone', 'SL_ISR08_2018.pdf',
     'https://documents.worldbank.org/curated/en/708651536244645345/pdf/Disclosable-Version-of-the-ISR-Sierra-Leone-Safety-Nets-Project-P143588-Sequence-No-08.pdf'),
    ('input', 'sierra-leone', 'SL_AF2_2019.pdf',
     'https://documents.worldbank.org/curated/en/342271562292188860/pdf/Sierra-Leone-Social-Safety-Net-Project-Second-Additional-Financing.pdf'),
    ('input', 'sierra-leone', 'SL_ISR14_2021.pdf',
     'https://documents.worldbank.org/curated/en/099410112212116875/pdf/Disclosable0Ve0588000Sequence0No014.pdf'),
    ('input', 'sierra-leone', 'SL_RESTRUCTURING_2023.pdf',
     'https://documents.worldbank.org/curated/en/099062023164040814/pdf/P1435880bb78fa01f091990677538ffcfd2.pdf'),
    ('input', 'sierra-leone', 'SL_ISR17_2023.pdf',
     'https://documents.worldbank.org/curated/en/099062623182521065/pdf/P14358805d8d7a0df08a45030315c8d7405.pdf'),
    ('holdout', 'sierra-leone', 'SL_ICR_HOLDOUT_2023.pdf',
     'https://documents.worldbank.org/curated/en/099122023184024702/pdf/BOSIB-cf2675ec-fe84-4259-9f9d-8e2d058c935c.pdf'),
    ('holdout', 'sierra-leone', 'SL_IEG_HOLDOUT_2024.pdf',
     'https://documents.worldbank.org/curated/en/099082025141523686/pdf/P143588-45175206-4e57-4db7-b96f-8d391ee27a57.pdf'),
)


def download_pdf(url: str, destination: Path) -> None:
    """Stream to a temporary file and replace the destination after validation."""
    if destination.suffix != ".pdf":
        raise ValueError("Destination filename must end in .pdf")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        # urlopen raises HTTPError for unsuccessful HTTP responses.
        request = Request(url, headers=REQUEST_HEADERS)
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, suffix=".part", delete=False
            ) as output:
                temporary_path = Path(output.name)
                shutil.copyfileobj(response, output, length=64 * 1024)
        with temporary_path.open("rb") as downloaded:
            signature = downloaded.read(4)
        if not signature:
            raise ValueError("Downloaded response is empty")
        if signature != b"%PDF":
            raise ValueError("Downloaded response does not begin with %PDF")
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Overwrite existing PDFs"
    )
    args = parser.parse_args()
    downloaded = skipped = failed = 0
    for split, country, filename, url in MANIFEST:
        relative_path = Path("data") / split / country / filename
        destination = REPO_ROOT / relative_path
        try:
            if destination.exists() and not args.force:
                print(f"Skipping existing file: {relative_path}")
                skipped += 1
                continue
            print(f"Downloading {filename} -> {relative_path}", flush=True)
            download_pdf(url, destination)
            downloaded += 1
        except Exception as error:
            # A failure for one document must not stop the rest of the corpus.
            print(f"Failed {relative_path}: {error}", file=sys.stderr)
            failed += 1
    print(
        f"Summary ({len(MANIFEST)} files): "
        f"downloaded={downloaded}, skipped={skipped}, failed={failed}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
