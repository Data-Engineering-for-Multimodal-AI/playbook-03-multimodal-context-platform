"""Generate the synthetic KYC sample documents in data/.

Every person, company, address, account and number here is invented. Social
Security numbers come from the 987-65-4320..4329 block that the Social Security
Administration reserves for advertising; employer EINs start with 00, a prefix
the IRS never assigns; phone numbers use the fictional 555-01xx range. Each
page is marked SPECIMEN.

The documents are consistent with each other, as a real customer's file would
be: W-2 wages and withholding flow into the Form 1040 (2024 tax table and
standard deduction), the bank statement shows the biweekly net pay from the
W-2 employer and the payment of the utility bill.

Output (22 documents, the layout the README's gsutil commands expect):
  data/DrivingLicenses/*.jpeg          5 driver's licenses (photos of a card)
  data/W2Sample/*W2.pdf                5 Forms W-2
  data/Form1040Sample/*Form1040.pdf    5 Forms 1040
  data/Utility/*.pdf                   6 utility bills
  data/Realistic_Bank_Statements.pdf   December 2024 statements for all five
  data/extracted_text/DrivingLicenses/*.txt
      the text printed on each card, used by local mode instead of OCR

Run:  uv run --group dev python scripts/generate_sample_data.py [--out data]
Needs Pillow and reportlab (the dev dependency group). Output is deterministic.
"""

import argparse
import math
import os
import random
from dataclasses import dataclass, field

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from reportlab.lib.colors import Color, HexColor, black, white
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas as rl_canvas

DISCLAIMER = ("SYNTHETIC SAMPLE - Data Engineering for Multimodal AI, Chapter 3. "
              "Not a real document. All names and numbers are fictitious.")
PERIODS = 26  # biweekly payroll


# --------------------------------------------------------------------------
# People
# --------------------------------------------------------------------------

@dataclass
class Person:
    first: str
    middle: str
    last: str
    initials: str          # driver's license file name, e.g. LRDL.jpeg
    file_stem: str         # e.g. LauraRusso
    ssn: str
    dob: str
    sex: str
    height: str
    weight: int
    eyes: str
    street: str
    city: str
    state: str
    state_name: str
    zip: str
    phone: str
    occupation: str
    employer: str
    employer_street: str
    employer_city: str
    ein: str
    state_employer_id: str
    gross_per_period: float
    retirement_pct: float
    retirement_code: str   # W-2 box 12 code: D = 401(k), E = 403(b)
    fed_wh_factor: float   # withholding relative to the actual tax
    filing_status: str     # "single" or "mfj"
    spouse_first: str = ""
    spouse_ssn: str = ""
    interest: float = 0.0
    dl_number: str = ""
    dl_class: str = "D"
    dl_issued: str = ""
    dl_expires: str = ""
    dl_dd: str = ""
    dl_color: str = "#1f4e8c"
    dl_title: str = "DRIVER'S LICENSE"
    rent_payee: str = ""
    rent: float = 0.0
    opening_balance: float = 0.0
    account_last4: str = ""
    utilities: list = field(default_factory=list)

    @property
    def full_name(self):
        return f"{self.first} {self.middle} {self.last}"

    @property
    def name(self):
        return f"{self.first} {self.last}"

    @property
    def city_line(self):
        return f"{self.city}, {self.state} {self.zip}"


@dataclass
class UtilityBill:
    file_name: str
    company: str
    company_street: str
    company_city: str
    service_phone: str
    kind: str              # "electric" or "water"
    account: str
    meter: str
    prev_read: int
    usage: int             # kWh, or thousand gallons for water
    history: list          # 13 months of usage, oldest first
    rates: dict
    pay_date: str          # date the bank statement shows the payment


PEOPLE = [
    Person(
        first="Laura", middle="Marie", last="Russo", initials="LR", file_stem="LauraRusso",
        ssn="987-65-4321", dob="04/17/1990", sex="F", height="5'-06\"", weight=135, eyes="BRN",
        street="2418 W Marlowe St Apt 3", city="Chicago", state="IL", state_name="ILLINOIS",
        zip="60618", phone="(312) 555-0142",
        occupation="Registered Nurse", employer="Halsted Park Medical Group LLC",
        employer_street="1700 N Halsted Park Ave", employer_city="Chicago, IL 60614",
        ein="00-4817263", state_employer_id="0481-7263",
        gross_per_period=3015.38, retirement_pct=0.05, retirement_code="D",
        fed_wh_factor=1.06, filing_status="single", interest=142.18,
        dl_number="R200-5418-7302", dl_issued="05/02/2025", dl_expires="04/17/2029",
        dl_dd="20250502R2005418", dl_color="#1d4f91",
        rent_payee="WESTRIDGE PROPERTY MGMT", rent=1685.00, opening_balance=4218.63,
        account_last4="4821",
        utilities=[UtilityBill(
            file_name="LauraRussoUtility.pdf", company="Prairie Lakes Energy",
            company_street="PO Box 81400", company_city="Chicago, IL 60681",
            service_phone="1-800-555-0110", kind="electric", account="4402 118 6375",
            meter="ELM 7731902", prev_read=48213, usage=512,
            history=[478, 455, 430, 402, 418, 466, 588, 702, 731, 655, 540, 497, 512],
            rates={"Customer charge": 15.98, "per_kwh": 0.0812, "delivery_per_kwh": 0.0569,
                   "taxes_pct": 0.0712},
            pay_date="12/09")],
    ),
    Person(
        first="Mitchell", middle="James", last="Allen", initials="MA", file_stem="MitchellAllen",
        ssn="987-65-4322", dob="09/03/1984", sex="M", height="6'-01\"", weight=188, eyes="BLU",
        street="517 Linden Hollow Rd", city="Pittsburgh", state="PA", state_name="PENNSYLVANIA",
        zip="15237", phone="(412) 555-0167",
        occupation="Software Engineer", employer="Three Rivers Data Systems Inc.",
        employer_street="200 Ferrous Way Suite 900", employer_city="Pittsburgh, PA 15222",
        ein="00-7731845", state_employer_id="77318450",
        gross_per_period=4788.46, retirement_pct=0.08, retirement_code="D",
        fed_wh_factor=0.97, filing_status="mfj", spouse_first="Rachel", spouse_ssn="987-65-4323",
        interest=611.40,
        dl_number="30 412 587", dl_class="C", dl_issued="09/12/2023", dl_expires="09/04/2027",
        dl_dd="0830412587 2023255", dl_color="#123a6b",
        rent_payee="KEYSTONE HOME LOANS MTG", rent=2140.27, opening_balance=9734.10,
        account_last4="7306",
        utilities=[UtilityBill(
            file_name="MitchellAllenUtility.pdf", company="Monongahela Hills Electric",
            company_street="PO Box 642180", company_city="Pittsburgh, PA 15264",
            service_phone="1-800-555-0123", kind="electric", account="1100 742 9018",
            meter="MHE 2290471", prev_read=91355, usage=884,
            history=[910, 862, 791, 702, 688, 845, 1104, 1230, 1186, 940, 802, 851, 884],
            rates={"Customer charge": 12.50, "per_kwh": 0.0975, "delivery_per_kwh": 0.0611,
                   "taxes_pct": 0.0590},
            pay_date="12/10")],
    ),
    Person(
        first="April", middle="Dawn", last="Hensley", initials="AH", file_stem="AprilHensley",
        ssn="987-65-4324", dob="12/22/1993", sex="F", height="5'-04\"", weight=126, eyes="GRN",
        street="3381 S Juniper Bluff Ct", city="Denver", state="CO", state_name="COLORADO",
        zip="80219", phone="(720) 555-0188",
        occupation="Elementary School Teacher", employer="Front Range Charter Academy",
        employer_street="4900 W Mesa Gate Blvd", employer_city="Denver, CO 80236",
        ein="00-2958804", state_employer_id="29588042",
        gross_per_period=2161.54, retirement_pct=0.04, retirement_code="E",
        fed_wh_factor=1.02, filing_status="single", interest=38.02,
        dl_number="17-308-4521", dl_class="R", dl_issued="01/09/2024", dl_expires="12/22/2028",
        dl_dd="2024009CO17308452100", dl_color="#2a5b3f", dl_title="DRIVER LICENSE",
        rent_payee="ASPEN TRAIL APARTMENTS", rent=1420.00, opening_balance=1904.55,
        account_last4="5530",
        utilities=[UtilityBill(
            file_name="AprilHensleyUtility.pdf", company="Front Range Gas & Electric",
            company_street="PO Box 30219", company_city="Denver, CO 80230",
            service_phone="1-800-555-0135", kind="electric", account="5310 004 2276",
            meter="FRG 0458812", prev_read=22964, usage=398,
            history=[421, 410, 377, 350, 362, 420, 530, 566, 544, 446, 389, 372, 398],
            rates={"Customer charge": 7.10, "per_kwh": 0.0889, "delivery_per_kwh": 0.0471,
                   "taxes_pct": 0.0645},
            pay_date="12/12")],
    ),
    Person(
        first="Taylor", middle="Lynn", last="Cox", initials="TC", file_stem="TaylorCox",
        ssn="987-65-4325", dob="06/30/1988", sex="F", height="5'-08\"", weight=150, eyes="HAZ",
        street="1209 Wrenfield Dr", city="Raleigh", state="NC", state_name="NORTH CAROLINA",
        zip="27610", phone="(919) 555-0151",
        occupation="Logistics Coordinator", employer="Piedmont Freight Partners LLC",
        employer_street="8800 Tobacco Barn Rd", employer_city="Raleigh, NC 27616",
        ein="00-5562190", state_employer_id="600556219",
        gross_per_period=2375.00, retirement_pct=0.03, retirement_code="D",
        fed_wh_factor=0.99, filing_status="single", interest=96.55,
        dl_number="000041738256", dl_class="C", dl_issued="07/15/2022", dl_expires="06/30/2030",
        dl_dd="0001735824915", dl_color="#7a1f2b", dl_title="DRIVERS LICENSE",
        rent_payee="OAKMONT RESIDENTIAL LLC", rent=1295.00, opening_balance=2687.41,
        account_last4="9147",
        utilities=[
            UtilityBill(
                file_name="Taylorcoxutility.pdf", company="Carolina Piedmont Power",
                company_street="PO Box 70144", company_city="Charlotte, NC 28272",
                service_phone="1-800-555-0147", kind="electric", account="9917 330 5482",
                meter="CPP 6620147", prev_read=60471, usage=946,
                history=[1012, 905, 820, 760, 810, 1102, 1388, 1460, 1395, 1050, 880, 902, 946],
                rates={"Customer charge": 14.00, "per_kwh": 0.0921, "delivery_per_kwh": 0.0388,
                       "taxes_pct": 0.0700},
                pay_date="12/11"),
            UtilityBill(
                file_name="TaylorCoxWaterUtility.pdf", company="Neuse River Water Authority",
                company_street="PO Box 50921", company_city="Raleigh, NC 27650",
                service_phone="1-800-555-0158", kind="water", account="20-118-4473-01",
                meter="NRW 304417", prev_read=1388, usage=5,
                history=[4, 4, 5, 4, 5, 6, 7, 7, 6, 5, 5, 4, 5],
                rates={"Water base charge": 7.86, "Sewer base charge": 11.27,
                       "water_per_kgal": 5.12, "sewer_per_kgal": 6.94,
                       "Stormwater fee": 6.40},
                pay_date="12/16"),
        ],
    ),
    Person(
        first="James", middle="Robert", last="Williams", initials="JW", file_stem="JamesWilliams",
        ssn="987-65-4326", dob="02/11/1979", sex="M", height="5'-11\"", weight=201, eyes="BRN",
        street="7046 Cedar Brake Ln", city="Austin", state="TX", state_name="TEXAS",
        zip="78745", phone="(512) 555-0176",
        occupation="Electrician", employer="Barton Ridge Electrical Services LLC",
        employer_street="12010 Caliche Creek Blvd", employer_city="Austin, TX 78748",
        ein="00-8840317", state_employer_id="",
        gross_per_period=2665.38, retirement_pct=0.0, retirement_code="",
        fed_wh_factor=1.04, filing_status="single", interest=12.67,
        dl_number="41827365", dl_class="C", dl_issued="03/01/2021", dl_expires="02/11/2029",
        dl_dd="41827365210601", dl_color="#3b3f8f", dl_title="DRIVER LICENSE",
        rent_payee="LIVE OAK RENTALS", rent=1510.00, opening_balance=3311.87,
        account_last4="2064",
        utilities=[UtilityBill(
            file_name="JamesWilliamsUtility.pdf", company="Hill Country Power & Light",
            company_street="PO Box 149060", company_city="Austin, TX 78714",
            service_phone="1-800-555-0169", kind="electric", account="6608 214 0937",
            meter="HCP 5108836", prev_read=118027, usage=1187,
            history=[1240, 1105, 960, 1010, 1330, 1702, 2015, 2240, 2110, 1630, 1280, 1150, 1187],
            rates={"Customer charge": 10.00, "per_kwh": 0.0744, "delivery_per_kwh": 0.0452,
                   "taxes_pct": 0.0425},
            pay_date="12/13")],
    ),
]


# --------------------------------------------------------------------------
# Tax and payroll arithmetic (tax year 2024)
# --------------------------------------------------------------------------

BRACKETS = {
    "single": [(11600, .10), (47150, .12), (100525, .22), (191950, .24),
               (243725, .32), (609350, .35), (math.inf, .37)],
    "mfj": [(23200, .10), (94300, .12), (201050, .22), (383900, .24),
            (487450, .32), (731200, .35), (math.inf, .37)],
}
STANDARD_DEDUCTION = {"single": 14600, "mfj": 29200}
SS_RATE, MEDICARE_RATE = 0.062, 0.0145


def bracket_tax(income, status):
    tax, lower = 0.0, 0.0
    for upper, rate in BRACKETS[status]:
        if income <= lower:
            break
        tax += (min(income, upper) - lower) * rate
        lower = upper
    return tax


def tax_2024(taxable, status):
    """Tax as the 2024 Tax Table gives it (below $100,000) or the worksheet (above)."""
    if taxable <= 0:
        return 0
    if taxable < 100000:
        width = 25 if taxable < 3000 else 50
        low = (taxable // width) * width
        return int(bracket_tax(low + width / 2, status) + 0.5)
    return int(bracket_tax(taxable, status) + 0.5)


def cents(x):
    return math.floor(x * 100 + 0.5) / 100


def money(x, sign=False):
    s = f"{abs(x):,.2f}"
    if x < 0:
        return f"-{s}"
    return f"+{s}" if sign else s


def dollars(x):
    return f"{int(round(x)):,}"


def state_tax_per_period(p, gross, box1_per):
    if p.state == "IL":   # 4.95% after the $2,775 exemption
        return cents((box1_per - 2775 / PERIODS) * 0.0495)
    if p.state == "PA":   # 3.07%; PA taxes 401(k) contributions
        return cents(gross * 0.0307)
    if p.state == "CO":   # 4.25% for 2024
        return cents(box1_per * 0.0425)
    if p.state == "NC":   # 4.5% after the $12,750 standard deduction
        return cents((box1_per - 12750 / PERIODS) * 0.045)
    return 0.0           # TX has no income tax


def payroll(p):
    gross = p.gross_per_period
    retire = cents(gross * p.retirement_pct)
    box1_per = gross - retire
    ss = cents(gross * SS_RATE)
    med = cents(gross * MEDICARE_RATE)
    box1 = cents(box1_per * PERIODS)
    taxable = box1 + p.interest - STANDARD_DEDUCTION[p.filing_status]
    fed_annual_target = bracket_tax(max(taxable, 0), p.filing_status) * p.fed_wh_factor
    fed = cents(fed_annual_target / PERIODS)
    st = state_tax_per_period(p, gross, box1_per)
    net = cents(gross - retire - fed - ss - med - st)
    per = dict(gross=gross, retire=retire, fed=fed, ss=ss, med=med, state=st, net=net)
    annual = {k: cents(v * PERIODS) for k, v in per.items()}
    annual["box1"] = box1
    annual["state_wages"] = cents(gross * PERIODS) if p.state == "PA" else box1
    return per, annual


def form_1040(p):
    _, w = payroll(p)
    l1a = round(w["box1"])
    l1z = l1a
    l2b = round(p.interest)
    l9 = l1z + l2b
    l11 = l9
    l12 = STANDARD_DEDUCTION[p.filing_status]
    l15 = max(l11 - l12, 0)
    l16 = tax_2024(l15, p.filing_status)
    l24 = l16
    l25a = round(w["fed"])
    l33 = l25a
    refund = max(l33 - l24, 0)
    owed = max(l24 - l33, 0)
    return dict(l1a=l1a, l1z=l1z, l2b=l2b, l9=l9, l11=l11, l12=l12, l15=l15, l16=l16,
                l24=l24, l25a=l25a, l25d=l25a, l33=l33, l34=refund, l35a=refund, l37=owed)


def utility_amounts(u):
    """Line items and total for one bill."""
    r = u.rates
    if u.kind == "electric":
        items = [
            ("Customer charge", r["Customer charge"]),
            (f"Energy charge  {u.usage:,} kWh x ${r['per_kwh']:.4f}", cents(u.usage * r["per_kwh"])),
            (f"Delivery charge  {u.usage:,} kWh x ${r['delivery_per_kwh']:.4f}",
             cents(u.usage * r["delivery_per_kwh"])),
        ]
        sub = sum(a for _, a in items)
        items.append((f"Taxes and fees ({r['taxes_pct'] * 100:.2f}%)", cents(sub * r["taxes_pct"])))
    else:
        items = [
            ("Water base charge", r["Water base charge"]),
            (f"Water usage  {u.usage} kgal x ${r['water_per_kgal']:.2f}",
             cents(u.usage * r["water_per_kgal"])),
            ("Sewer base charge", r["Sewer base charge"]),
            (f"Sewer usage  {u.usage} kgal x ${r['sewer_per_kgal']:.2f}",
             cents(u.usage * r["sewer_per_kgal"])),
            ("Stormwater fee", r["Stormwater fee"]),
        ]
    return items, cents(sum(a for _, a in items))


# --------------------------------------------------------------------------
# PDF helpers
# --------------------------------------------------------------------------

def new_pdf(path, title):
    c = rl_canvas.Canvas(path, pagesize=letter, invariant=1)
    c.setTitle(title)
    c.setAuthor("Synthetic sample data")
    c.setCreator("generate_sample_data.py")
    return c


def specimen_marks(c):
    w, h = letter
    c.saveState()
    c.setFillColor(Color(0.75, 0.1, 0.1, alpha=0.10))
    c.setFont("Helvetica-Bold", 96)
    c.translate(w / 2, h / 2)
    c.rotate(35)
    c.drawCentredString(0, 0, "SPECIMEN")
    c.restoreState()
    c.setFont("Helvetica", 6.5)
    c.setFillColor(HexColor("#666666"))
    c.drawCentredString(w / 2, 18, DISCLAIMER)
    c.setFillColor(black)


def box(c, x, y, w, h, label, value="", label_size=6, value_size=9.5, align="left", bold=False):
    c.setLineWidth(0.6)
    c.rect(x, y, w, h)
    c.setFont("Helvetica", label_size)
    c.drawString(x + 2.5, y + h - label_size - 1.5, label)
    if value != "":
        c.setFont("Helvetica-Bold" if bold else "Helvetica", value_size)
        lines = value.split("\n")
        for i, line in enumerate(lines):
            vy = y + h - label_size - 4 - (i + 1) * (value_size + 1.5)
            if align == "right":
                c.drawRightString(x + w - 4, vy, line)
            else:
                c.drawString(x + 6, vy, line)


# --------------------------------------------------------------------------
# Form W-2
# --------------------------------------------------------------------------

def make_w2(p, path):
    per, a = payroll(p)
    c = new_pdf(path, f"Form W-2 2024 - {p.name}")
    W, H = letter
    x0, top = 36, H - 60

    # Title block (printed at the bottom of the form, drawn first so text extraction starts with it)
    yb = top - 9 * 30 - 110
    c.setFont("Helvetica-Bold", 22)
    c.drawString(x0, yb, "W-2")
    c.setFont("Helvetica-Bold", 13)
    c.drawString(x0 + 50, yb + 2, "Wage and Tax Statement")
    c.setFont("Helvetica-Bold", 22)
    c.drawString(x0 + 300, yb, "2024")
    c.setFont("Helvetica", 7.5)
    c.drawRightString(x0 + 540, yb + 10, "Department of the Treasury-Internal Revenue Service")
    c.drawString(x0, yb - 18, "Form W-2 Wage and Tax Statement 2024. Copy B-To Be Filed With Employee's "
                 "FEDERAL Tax Return.")
    c.drawString(x0, yb - 30, f"Employee: {p.first} {p.middle} {p.last}.  Employer: {p.employer}.")


    c.setFont("Helvetica-Bold", 9)
    c.drawString(x0, top + 22, "Copy B-To Be Filed With Employee's FEDERAL Tax Return.")
    c.setFont("Helvetica", 7)
    c.drawString(x0, top + 12, "This information is being furnished to the Internal Revenue Service.")

    lw, rw, rh = 270, 270, 30
    xr = x0 + lw
    # Left column
    box(c, x0, top - rh, 130, rh, "a  Employee's social security number", p.ssn)
    box(c, x0 + 130, top - rh, 140, rh, "OMB No. 1545-0008", "", label_size=7)
    box(c, x0, top - 2 * rh, lw, rh, "b  Employer identification number (EIN)", p.ein)
    box(c, x0, top - 2 * rh - 64, lw, 64, "c  Employer's name, address, and ZIP code",
        f"{p.employer}\n{p.employer_street}\n{p.employer_city}")
    box(c, x0, top - 3 * rh - 64, lw, rh, "d  Control number", f"{p.initials}-2024-{p.ssn[-4:]}")
    box(c, x0, top - 3 * rh - 64 - 74, lw, 74, "e  Employee's first name and initial   Last name   Suff.",
        f"{p.first} {p.middle[0]}  {p.last}\n{p.street}\n{p.city_line}")
    e_bottom = top - 3 * rh - 64 - 74
    box(c, x0, top - 9 * rh, lw, e_bottom - (top - 9 * rh), "f  Employee's address and ZIP code")

    # Right column, boxes 1-14
    rows = [
        ("1  Wages, tips, other compensation", money(a["box1"]),
         "2  Federal income tax withheld", money(a["fed"])),
        ("3  Social security wages", money(a["gross"]), "4  Social security tax withheld", money(a["ss"])),
        ("5  Medicare wages and tips", money(a["gross"]), "6  Medicare tax withheld", money(a["med"])),
        ("7  Social security tips", "", "8  Allocated tips", ""),
        ("9", "", "10  Dependent care benefits", ""),
        ("11  Nonqualified plans", "",
         "12a  See instructions for box 12",
         f"{p.retirement_code}   {money(a['retire'])}" if p.retirement_code else ""),
    ]
    y = top - rh
    for l1, v1, l2, v2 in rows:
        box(c, xr, y, rw / 2, rh, l1, v1, align="right")
        box(c, xr + rw / 2, y, rw / 2, rh, l2, v2, align="right")
        y -= rh
    box(c, xr, y - rh, rw / 2, rh * 2, "13  Statutory empl.  Retirement plan  Sick pay",
        "            X" if p.retirement_code else "", label_size=5.5)
    box(c, xr + rw / 2, y, rw / 2, rh, "12b", "")
    box(c, xr + rw / 2, y - rh, rw / 2, rh, "12c", "")
    box(c, xr, y - 2 * rh, rw, rh, "14  Other", "")
    y_state = y - 2 * rh - 40
    c.setLineWidth(0.6)
    cols = [(x0, 40, "15  State"), (x0 + 40, 130, "Employer's state ID number"),
            (x0 + 170, 100, "16  State wages, tips, etc."), (x0 + 270, 90, "17  State income tax"),
            (x0 + 360, 90, "18  Local wages, tips, etc."), (x0 + 450, 90, "19  Local income tax")]
    vals = ([p.state, p.state_employer_id, money(a["state_wages"]), money(a["state"]), "", ""]
            if p.state != "TX" else ["", "", "", "", "", ""])
    for (cx, cw, lab), v in zip(cols, vals):
        box(c, cx, y_state, cw, 40, lab, v, align="right" if cx > x0 + 100 else "left")
    box(c, x0 + 450, y_state - 24, 90, 24, "20  Locality name", "")

    specimen_marks(c)
    c.showPage()
    c.save()


# --------------------------------------------------------------------------
# Form 1040
# --------------------------------------------------------------------------

def line(c, y, num, text, amount, x=40, right=572, num_x=470):
    c.setFont("Helvetica-Bold", 8)
    c.drawString(x, y, num)
    c.setFont("Helvetica", 8)
    c.drawString(x + 24, y, text)
    c.setDash(1, 2)
    c.line(x + 26 + c.stringWidth(text, "Helvetica", 8), y - 1, num_x - 4, y - 1)
    c.setDash()
    c.rect(num_x, y - 4, 22, 13)
    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(num_x + 11, y, num)
    c.rect(num_x + 22, y - 4, right - num_x - 22, 13)
    c.setFont("Helvetica", 9)
    if amount != "":
        c.drawRightString(right - 4, y, amount)


def make_1040(p, path):
    f = form_1040(p)
    c = new_pdf(path, f"Form 1040 2024 - {p.name}")
    W, H = letter
    x, y = 40, H - 50

    # Page 1 header
    c.setFont("Helvetica-Bold", 22)
    c.drawString(x, y - 6, "1040")
    c.setFont("Helvetica", 7)
    c.drawString(x, y + 12, "Form")
    c.setFont("Helvetica", 8)
    c.drawString(x + 58, y + 8, "Department of the Treasury-Internal Revenue Service")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x + 58, y - 6, "U.S. Individual Income Tax Return")
    c.setFont("Helvetica-Bold", 20)
    c.drawString(x + 300, y - 6, "2024")
    c.setFont("Helvetica", 7)
    c.drawString(x + 362, y - 2, "OMB No. 1545-0074")
    c.drawString(x + 362, y - 11, "IRS Use Only-Do not write or staple in this space.")
    c.line(x, y - 14, 572, y - 14)
    c.setFont("Helvetica", 7.5)
    c.drawString(x, y - 26, "For the year Jan. 1-Dec. 31, 2024, or other tax year beginning ______, 2024, "
                 "ending ______, 20__")

    y -= 40
    rh = 24
    spouse = p.filing_status == "mfj"
    box(c, x, y - rh, 230, rh, "Your first name and middle initial", f"{p.first} {p.middle[0]}")
    box(c, x + 230, y - rh, 162, rh, "Last name", p.last)
    box(c, x + 392, y - rh, 140, rh, "Your social security number", p.ssn)
    y -= rh
    box(c, x, y - rh, 230, rh, "If joint return, spouse's first name and middle initial",
        p.spouse_first if spouse else "")
    box(c, x + 230, y - rh, 162, rh, "Last name", p.last if spouse else "")
    box(c, x + 392, y - rh, 140, rh, "Spouse's social security number", p.spouse_ssn if spouse else "")
    y -= rh
    street, apt = (p.street.rsplit(" Apt ", 1) + [""])[:2]
    box(c, x, y - rh, 340, rh, "Home address (number and street). If you have a P.O. box, see instructions.",
        street)
    box(c, x + 340, y - rh, 52, rh, "Apt. no.", apt)
    box(c, x + 392, y - rh * 2, 140, rh * 2, "Presidential Election Campaign",
        "Check here if you, or your\nspouse if filing jointly, want\n$3 to go to this fund.", value_size=6.5)
    y -= rh
    box(c, x, y - rh, 230, rh, "City, town, or post office", p.city)
    box(c, x + 230, y - rh, 60, rh, "State", p.state)
    box(c, x + 290, y - rh, 102, rh, "ZIP code", p.zip)
    y -= rh + 14

    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(x, y, "Filing Status")
    statuses = ["Single", "Married filing jointly (even if only one had income)",
                "Married filing separately (MFS)", "Head of household (HOH)",
                "Qualifying surviving spouse (QSS)"]
    chosen = 0 if p.filing_status == "single" else 1
    c.setFont("Helvetica", 8)
    sx, sy = x + 70, y
    for i, s in enumerate(statuses):
        bx = sx + (i % 2) * 230
        by = sy - (i // 2) * 12
        c.rect(bx, by - 1, 7, 7)
        if i == chosen:
            c.drawString(bx + 1.2, by, "X")
        c.drawString(bx + 11, by, s)
    y -= 46
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(x, y, "Digital Assets")
    c.setFont("Helvetica", 8)
    c.drawString(x + 70, y, "At any time during 2024, did you: (a) receive (as a reward, award, or payment for "
                 "property or services); or (b) sell,")
    c.drawString(x + 70, y - 10, "exchange, or otherwise dispose of a digital asset (or a financial interest in a "
                 "digital asset)?       Yes   [X] No")
    y -= 26
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(x, y, "Standard Deduction")
    c.setFont("Helvetica", 8)
    c.drawString(x + 90, y, "Someone can claim:  [ ] You as a dependent   [ ] Your spouse as a dependent")
    y -= 14
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(x, y, "Dependents")
    c.setFont("Helvetica", 8)
    c.drawString(x + 90, y, "(see instructions)   None")
    y -= 22

    c.setFont("Helvetica-Bold", 9)
    c.drawString(x, y + 10, "Income")
    rows1 = [
        ("1a", "Total amount from Form(s) W-2, box 1 (see instructions)", dollars(f["l1a"])),
        ("1b", "Household employee wages not reported on Form(s) W-2", ""),
        ("1c", "Tip income not reported on line 1a (see instructions)", ""),
        ("1z", "Add lines 1a through 1h", dollars(f["l1z"])),
        ("2b", "Taxable interest (2a Tax-exempt interest: 0)", dollars(f["l2b"])),
        ("3b", "Ordinary dividends (3a Qualified dividends: 0)", "0"),
        ("4b", "IRA distributions, taxable amount", ""),
        ("5b", "Pensions and annuities, taxable amount", ""),
        ("6b", "Social security benefits, taxable amount", ""),
        ("7", "Capital gain or (loss). Attach Schedule D if required", ""),
        ("8", "Additional income from Schedule 1, line 10", ""),
        ("9", "Add lines 1z, 2b, 3b, 4b, 5b, 6b, 7, and 8. This is your total income", dollars(f["l9"])),
        ("10", "Adjustments to income from Schedule 1, line 26", ""),
        ("11", "Subtract line 10 from line 9. This is your adjusted gross income", dollars(f["l11"])),
        ("12", "Standard deduction or itemized deductions (from Schedule A)", dollars(f["l12"])),
        ("13", "Qualified business income deduction from Form 8995 or Form 8995-A", ""),
        ("14", "Add lines 12 and 13", dollars(f["l12"])),
        ("15", "Subtract line 14 from line 11. If zero or less, enter -0-. This is your taxable income",
         dollars(f["l15"])),
    ]
    for num, text, amt in rows1:
        line(c, y, num, text, amt)
        y -= 15
    c.setFont("Helvetica", 7)
    c.drawString(x, y - 4, "For Disclosure, Privacy Act, and Paperwork Reduction Act Notice, see separate "
                 "instructions.        Cat. No. 11320B        Form 1040 (2024)")
    specimen_marks(c)
    c.showPage()

    # Page 2
    y = H - 50
    c.setFont("Helvetica", 8)
    c.drawString(x, y, "Form 1040 (2024)")
    c.drawRightString(572, y, "Page 2")
    c.line(x, y - 4, 572, y - 4)
    y -= 22
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x, y + 10, "Tax and Credits")
    rows2 = [
        ("16", "Tax (see instructions). Check if any from Form(s): 8814  4972", dollars(f["l16"])),
        ("17", "Amount from Schedule 2, line 3", ""),
        ("18", "Add lines 16 and 17", dollars(f["l16"])),
        ("19", "Child tax credit or credit for other dependents from Schedule 8812", ""),
        ("21", "Add lines 19 and 20", "0"),
        ("22", "Subtract line 21 from line 18. If zero or less, enter -0-", dollars(f["l16"])),
        ("23", "Other taxes, including self-employment tax, from Schedule 2, line 21", ""),
        ("24", "Add lines 22 and 23. This is your total tax", dollars(f["l24"])),
    ]
    for num, text, amt in rows2:
        line(c, y, num, text, amt)
        y -= 15
    y -= 6
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x, y + 10, "Payments")
    rows3 = [
        ("25a", "Federal income tax withheld from Form(s) W-2", dollars(f["l25a"])),
        ("25b", "Federal income tax withheld from Form(s) 1099", ""),
        ("25d", "Add lines 25a through 25c", dollars(f["l25d"])),
        ("26", "2024 estimated tax payments and amount applied from 2023 return", ""),
        ("33", "Add lines 25d, 26, and 32. These are your total payments", dollars(f["l33"])),
    ]
    for num, text, amt in rows3:
        line(c, y, num, text, amt)
        y -= 15
    y -= 6
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x, y + 10, "Refund")
    rows4 = [
        ("34", "If line 33 is more than line 24, subtract line 24 from line 33. This is the amount you "
               "overpaid", dollars(f["l34"]) if f["l34"] else ""),
        ("35a", "Amount of line 34 you want refunded to you", dollars(f["l35a"]) if f["l35a"] else ""),
    ]
    for num, text, amt in rows4:
        line(c, y, num, text, amt)
        y -= 15
    if f["l35a"]:
        c.setFont("Helvetica", 8)
        c.drawString(x + 24, y, f"b Routing number  XXXXXXXXX   c Type: [X] Checking   "
                                f"d Account number  XXXXXX{p.account_last4}")
        y -= 15
    y -= 6
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x, y + 10, "Amount You Owe")
    line(c, y, "37", "Subtract line 33 from line 24. This is the amount you owe",
         dollars(f["l37"]) if f["l37"] else "")
    y -= 30

    c.setFont("Helvetica-Bold", 9)
    c.drawString(x, y, "Sign Here")
    c.setFont("Helvetica", 7)
    c.drawString(x + 60, y, "Under penalties of perjury, I declare that I have examined this return and "
                 "accompanying schedules and statements, and to the")
    c.drawString(x + 60, y - 9, "best of my knowledge and belief, they are true, correct, and complete.")
    y -= 22
    box(c, x + 60, y - 24, 170, 24, "Your signature", f"{p.first} {p.last}")
    box(c, x + 230, y - 24, 70, 24, "Date", "03/14/2025")
    box(c, x + 300, y - 24, 140, 24, "Your occupation", p.occupation)
    box(c, x + 440, y - 24, 92, 24, "Identity Protection PIN", "")
    if p.filing_status == "mfj":
        y -= 24
        box(c, x + 60, y - 24, 170, 24, "Spouse's signature", f"{p.spouse_first} {p.last}")
        box(c, x + 230, y - 24, 70, 24, "Date", "03/14/2025")
        box(c, x + 300, y - 24, 140, 24, "Spouse's occupation", "Homemaker")
    y -= 30
    box(c, x + 60, y - 24, 200, 24, "Phone no.", p.phone)
    box(c, x + 260, y - 24, 272, 24, "Email address", "")
    y -= 40
    c.setFont("Helvetica", 7.5)
    c.drawString(x, y, "Paid Preparer Use Only:  Self-prepared")
    c.drawString(x, y - 24, "Go to www.irs.gov/Form1040 for instructions and the latest information.")
    c.drawRightString(572, y - 24, "Form 1040 (2024)")
    specimen_marks(c)
    c.showPage()
    c.save()


# --------------------------------------------------------------------------
# Utility bill
# --------------------------------------------------------------------------

MONTHS = ["Nov", "Dec", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov"]


def make_utility(p, u, path):
    items, total = utility_amounts(u)
    unit = "kWh" if u.kind == "electric" else "kgal"
    c = new_pdf(path, f"{u.company} statement - {p.name}")
    W, H = letter
    x, y = 40, H - 54

    c.setFillColor(HexColor("#1b5e7a") if u.kind == "electric" else HexColor("#1f6f5c"))
    c.rect(0, H - 80, W, 80, fill=1, stroke=0)
    c.setFillColor(white)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(x, y, u.company)
    c.setFont("Helvetica", 9)
    c.drawString(x, y - 16, f"{u.company_street}, {u.company_city}   Customer service {u.service_phone}")
    c.setFont("Helvetica-Bold", 11)
    c.drawRightString(W - 40, y, "Electric Service Statement" if u.kind == "electric"
                      else "Water and Sewer Statement")
    c.setFillColor(black)

    y = H - 110
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x, y, p.full_name.upper())
    c.setFont("Helvetica", 10)
    c.drawString(x, y - 13, p.street.upper())
    c.drawString(x, y - 26, p.city_line.upper())

    info = [("Account number", u.account), ("Statement date", "November 18, 2024"),
            ("Billing period", "10/15/2024 - 11/14/2024 (31 days)"),
            ("Service address", p.street)]
    iy = y
    for k, v in info:
        c.setFont("Helvetica", 8.5)
        c.drawString(310, iy, k)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawString(400, iy, v)
        iy -= 13

    # Amount due panel
    y -= 62
    c.setFillColor(HexColor("#eef3f6"))
    c.rect(x, y - 52, W - 80, 52, fill=1, stroke=0)
    c.setFillColor(black)
    c.setFont("Helvetica", 9)
    c.drawString(x + 10, y - 16, "Previous balance")
    c.drawRightString(x + 230, y - 16, f"${money(u.rates.get('_prev', total * 0.97))}")
    c.drawString(x + 10, y - 30, "Payment received Oct 28 - thank you")
    c.drawRightString(x + 230, y - 30, f"-${money(u.rates.get('_prev', total * 0.97))}")
    c.drawString(x + 10, y - 44, "Current charges")
    c.drawRightString(x + 230, y - 44, f"${money(total)}")
    c.setFont("Helvetica-Bold", 12)
    c.drawString(x + 290, y - 22, "Amount due by Dec 9, 2024")
    c.setFont("Helvetica-Bold", 18)
    c.drawRightString(W - 50, y - 24, f"${money(total)}")
    c.setFont("Helvetica", 8)
    c.drawString(x + 290, y - 40, "AutoPay: enrolled. This amount will be drafted from your bank account.")

    # Meter
    y -= 80
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x, y, "Meter reading")
    y -= 14
    c.setFont("Helvetica", 8.5)
    hdr = ["Meter", "Read dates", "Previous", "Current", "Multiplier", f"Usage ({unit})"]
    cur = u.prev_read + u.usage
    vals = [u.meter, "10/15 - 11/14", f"{u.prev_read:,}", f"{cur:,}", "1", f"{u.usage:,}"]
    xs = [x, x + 90, x + 190, x + 270, x + 350, x + 430]
    for xx, h_ in zip(xs, hdr):
        c.drawString(xx, y, h_)
    c.line(x, y - 3, W - 40, y - 3)
    c.setFont("Helvetica-Bold", 8.5)
    for xx, v in zip(xs, vals):
        c.drawString(xx, y - 14, v)

    # Charges
    y -= 42
    c.setFont("Helvetica-Bold", 10)
    c.drawString(x, y, "Current charges")
    y -= 14
    c.setFont("Helvetica", 9)
    for label, amt in items:
        c.drawString(x + 10, y, label)
        c.drawRightString(x + 330, y, money(amt))
        y -= 13
    c.line(x + 10, y + 8, x + 330, y + 8)
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x + 10, y - 3, "Total current charges")
    c.drawRightString(x + 330, y - 3, money(total))

    # Usage history chart
    cx, cy, cw, ch = x, y - 150, 330, 95
    c.setFont("Helvetica-Bold", 9)
    c.drawString(cx, cy + ch + 18, f"Your usage history ({unit} per month)")
    mx = max(u.history)
    bw = cw / len(u.history)
    for i, v in enumerate(u.history):
        bh = ch * v / mx
        c.setFillColor(HexColor("#7fa7bd") if i < len(u.history) - 1 else HexColor("#1b5e7a"))
        c.rect(cx + i * bw + 1, cy, bw - 2, bh, fill=1, stroke=0)
        c.setFillColor(black)
        c.setFont("Helvetica", 5.5)
        c.drawCentredString(cx + i * bw + bw / 2, cy - 7, MONTHS[i])
    avg = sum(u.history) / len(u.history)
    c.setFont("Helvetica", 7.5)
    c.drawString(cx, cy - 20, f"This period: {u.usage:,} {unit}   12-month average: {avg:,.0f} {unit}")

    # Remittance stub
    y = 150
    c.setDash(3, 3)
    c.line(20, y + 20, W - 20, y + 20)
    c.setDash()
    c.setFont("Helvetica", 7)
    c.drawString(x, y + 24, "Please detach and return this portion with your payment. "
                 "Make checks payable to " + u.company + ".")
    c.setFont("Helvetica-Bold", 9)
    c.drawString(x, y, u.company)
    c.setFont("Helvetica", 9)
    c.drawString(x, y - 12, u.company_street)
    c.drawString(x, y - 24, u.company_city)
    stub = [("Account number", u.account), ("Due date", "12/09/2024"),
            ("Amount due", f"${money(total)}"), ("Amount enclosed", "$__________")]
    sy = y
    for k, v in stub:
        c.setFont("Helvetica", 8.5)
        c.drawString(330, sy, k)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawRightString(W - 40, sy, v)
        sy -= 14
    c.setFont("Helvetica", 9)
    c.drawString(x, y - 52, p.full_name.upper())
    c.drawString(x, y - 64, p.street.upper())
    c.drawString(x, y - 76, p.city_line.upper())
    c.setFont("Courier", 9)
    c.drawString(330, y - 76, f"{u.account.replace(' ', '').replace('-', '')}  "
                              f"{int(total * 100):08d}  2")
    specimen_marks(c)
    c.showPage()
    c.save()
    return total


# --------------------------------------------------------------------------
# Bank statements
# --------------------------------------------------------------------------

BANK = "Cedar Plains Bank"
BANK_ADDR = "PO Box 4120, Wichita, KS 67201"
BANK_PHONE = "1-800-555-0191"


def bank_transactions(p, rng):
    per, _ = payroll(p)
    employer_tag = p.employer.upper().replace(".", "").replace(",", "")[:28]
    tx = [("12/02", f"ACH DEBIT {p.rent_payee} RENT/MTG", -p.rent)]
    for u in p.utilities:
        _, amt = utility_amounts(u)
        tx.append((u.pay_date, f"ACH DEBIT {u.company.upper()} AUTOPAY {u.account[-4:]}", -amt))
    tx += [
        ("12/06", f"DIRECT DEP {employer_tag} PAYROLL", per["net"]),
        ("12/20", f"DIRECT DEP {employer_tag} PAYROLL", per["net"]),
        ("12/31", "INTEREST PAYMENT", cents(p.opening_balance * 0.0001 + 0.05)),
    ]
    merchants = [("GREENLEAF MARKET #2", (48, 162)), ("FUEL STOP #0412", (31, 58)),
                 ("CORNER PHARMACY", (9, 41)), ("DAILY GRIND COFFEE", (4, 13)),
                 ("HOMEWARE OUTLET", (22, 95)), ("CITY TRANSIT FARE", (2.5, 2.5)),
                 ("NORTHSIDE DINER", (18, 64))]
    for day in [3, 5, 8, 9, 12, 14, 16, 18, 21, 23, 27, 29]:
        name, (lo, hi) = merchants[rng.randrange(len(merchants))]
        amt = cents(rng.uniform(lo, hi))
        tx.append((f"12/{day:02d}", f"DEBIT CARD PURCHASE {name} {p.city.upper()} {p.state}", -amt))
    tx.append(("12/15", "ATM WITHDRAWAL", -float(rng.choice([60, 80, 100, 120]))))
    tx.append(("12/17", "ACH DEBIT WIRELESS SVC AUTOPAY", -cents(rng.uniform(55, 85))))
    tx.append(("12/22", f"ONLINE TRANSFER TO SAVINGS XXXXXX{int(p.account_last4) + 1:04d}",
               -float(rng.choice([150, 200, 250, 300]))))
    tx.sort(key=lambda t: t[0])
    return tx


def make_bank_statements(path):
    c = new_pdf(path, f"{BANK} statements - December 2024")
    W, H = letter
    rng = random.Random(20241231)
    for p in PEOPLE:
        tx = bank_transactions(p, rng)
        deposits = cents(sum(a for _, _, a in tx if a > 0))
        withdrawals = cents(-sum(a for _, _, a in tx if a < 0))
        ending = cents(p.opening_balance + deposits - withdrawals)
        x, y = 40, H - 54

        c.setFillColor(HexColor("#24456b"))
        c.rect(0, H - 76, W, 76, fill=1, stroke=0)
        c.setFillColor(white)
        c.setFont("Helvetica-Bold", 20)
        c.drawString(x, y, BANK)
        c.setFont("Helvetica", 9)
        c.drawString(x, y - 15, f"{BANK_ADDR}   Customer service {BANK_PHONE}   Member FDIC")
        c.setFont("Helvetica-Bold", 11)
        c.drawRightString(W - 40, y, "Personal Checking Statement")
        c.setFillColor(black)

        y = H - 106
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x, y, p.full_name.upper())
        c.setFont("Helvetica", 10)
        c.drawString(x, y - 13, p.street.upper())
        c.drawString(x, y - 26, p.city_line.upper())
        for i, (k, v) in enumerate([("Statement period", "December 1 - December 31, 2024"),
                                    ("Account", f"Everyday Checking XXXXXX{p.account_last4}"),
                                    ("Account holder", p.name)]):
            c.setFont("Helvetica", 8.5)
            c.drawString(320, y - 13 * i, k)
            c.setFont("Helvetica-Bold", 8.5)
            c.drawString(410, y - 13 * i, v)

        y -= 54
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x, y, "Account summary")
        y -= 6
        summary = [("Beginning balance on December 1, 2024", money(p.opening_balance)),
                   ("Deposits and other credits", money(deposits)),
                   ("Withdrawals and other debits", money(withdrawals)),
                   ("Ending balance on December 31, 2024", money(ending))]
        for k, v in summary:
            y -= 13
            c.setFont("Helvetica-Bold" if k.startswith("Ending") else "Helvetica", 9)
            c.drawString(x + 10, y, k)
            c.drawRightString(x + 330, y, f"${v}" if k[0] in "BE" else f"{'+' if 'Dep' in k else '-'}${v}")

        y -= 30
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x, y, "Transaction history")
        y -= 16
        c.setFont("Helvetica-Bold", 8)
        c.drawString(x, y, "Date")
        c.drawString(x + 40, y, "Description")
        c.drawRightString(x + 400, y, "Deposits")
        c.drawRightString(x + 460, y, "Withdrawals")
        c.drawRightString(W - 40, y, "Balance")
        c.line(x, y - 4, W - 40, y - 4)
        bal = p.opening_balance
        c.setFont("Helvetica", 8)
        for d, desc, amt in tx:
            y -= 13
            bal = cents(bal + amt)
            c.drawString(x, y, d)
            c.drawString(x + 40, y, desc[:62])
            if amt > 0:
                c.drawRightString(x + 400, y, money(amt))
            else:
                c.drawRightString(x + 460, y, money(-amt))
            c.drawRightString(W - 40, y, money(bal))
        assert abs(bal - ending) < 0.005
        y -= 24
        c.setFont("Helvetica", 7.5)
        c.drawString(x, y, f"Linked Savings XXXXXX{int(p.account_last4) + 1:04d}: interest paid in 2024 "
                           f"${money(p.interest)}.  Checking APY earned this period: 0.01%.")
        c.drawString(x, y - 11, "In case of errors or questions about your electronic transfers, call "
                     f"{BANK_PHONE} as soon as you can.")
        specimen_marks(c)
        c.showPage()
    c.save()


# --------------------------------------------------------------------------
# Driver's license (rendered card, then "photographed" on a desk)
# --------------------------------------------------------------------------

def font(size):
    return ImageFont.load_default(size=size)


def dl_lines(p):
    """The text printed on the card, in reading order."""
    name1 = p.last.upper()
    name2 = f"{p.first.upper()} {p.middle.upper()}"
    return [
        f"{p.state_name}",
        f"{p.dl_title}",
        "USA",
        f"4d DLN {p.dl_number}",
        f"3 DOB {p.dob}",
        f"4b EXP {p.dl_expires}",
        f"1 {name1}",
        f"2 {name2}",
        f"8 {p.street.upper()}",
        f"{p.city.upper()}, {p.state} {p.zip}",
        f"4a ISS {p.dl_issued}",
        f"9 CLASS {p.dl_class}",
        "9a END NONE",
        "12 REST NONE",
        f"15 SEX {p.sex}",
        f"16 HGT {p.height}",
        f"17 WGT {p.weight} lb",
        f"18 EYES {p.eyes}",
        f"5 DD {p.dl_dd}",
        "SPECIMEN",
        "SAMPLE - NOT A GOVERNMENT DOCUMENT",
    ]


def render_card(p):
    W, H = 1012, 638
    base = HexColor(p.dl_color).rgb()
    rgb = tuple(int(v * 255) for v in base)
    card = Image.new("RGB", (W, H), (240, 242, 238))
    d = ImageDraw.Draw(card)
    # Background guilloche-like waves
    for k in range(0, H, 9):
        pts = [(xx, k + 6 * math.sin(xx / 37 + k / 23)) for xx in range(0, W, 8)]
        d.line(pts, fill=(222, 228, 226), width=1)
    d.rectangle([0, 0, W, 108], fill=rgb)
    d.text((34, 18), p.state_name, font=font(52), fill="white", stroke_width=1, stroke_fill="white")
    d.text((W - 34, 24), p.dl_title, font=font(30), fill="white", anchor="ra")
    d.text((W - 34, 66), "USA", font=font(24), fill="white", anchor="ra")

    # Portrait placeholder (silhouette; no real face)
    px, py, pw, ph = 40, 140, 270, 340
    d.rectangle([px, py, px + pw, py + ph], fill=(205, 212, 218))
    d.ellipse([px + 85, py + 55, px + 185, py + 175], fill=(150, 160, 170))
    d.pieslice([px + 30, py + 175, px + 240, py + 420], 180, 360, fill=(150, 160, 170))
    d.text((px + pw / 2, py + ph + 10), "PHOTO", font=font(16), fill=(110, 110, 110), anchor="ma")

    blue = (25, 35, 60)
    red = (165, 25, 35)
    fx = 345
    d.text((fx, 128), "4d DLN", font=font(20), fill=blue)
    d.text((fx + 90, 120), p.dl_number, font=font(34), fill=red, stroke_width=1, stroke_fill=red)
    d.text((fx, 172), "3 DOB", font=font(20), fill=blue)
    d.text((fx + 90, 166), p.dob, font=font(30), fill=red)
    d.text((fx + 330, 172), "4b EXP", font=font(20), fill=blue)
    d.text((fx + 410, 166), p.dl_expires, font=font(28), fill=blue)
    d.text((fx, 218), "1", font=font(20), fill=blue)
    d.text((fx + 30, 210), p.last.upper(), font=font(34), fill=blue, stroke_width=1, stroke_fill=blue)
    d.text((fx, 258), "2", font=font(20), fill=blue)
    d.text((fx + 30, 252), f"{p.first.upper()} {p.middle.upper()}", font=font(30), fill=blue)
    d.text((fx, 302), "8", font=font(20), fill=blue)
    d.text((fx + 30, 298), p.street.upper(), font=font(24), fill=blue)
    d.text((fx + 30, 328), f"{p.city.upper()}, {p.state} {p.zip}", font=font(24), fill=blue)
    d.text((fx, 372), "4a ISS", font=font(18), fill=blue)
    d.text((fx + 70, 368), p.dl_issued, font=font(24), fill=blue)
    d.text((fx + 230, 372), "9 CLASS", font=font(18), fill=blue)
    d.text((fx + 310, 368), p.dl_class, font=font(24), fill=blue)
    d.text((fx + 360, 372), "9a END", font=font(18), fill=blue)
    d.text((fx + 430, 368), "NONE", font=font(24), fill=blue)
    d.text((fx, 410), "12 REST", font=font(18), fill=blue)
    d.text((fx + 80, 406), "NONE", font=font(24), fill=blue)
    d.text((fx + 230, 410), "15 SEX", font=font(18), fill=blue)
    d.text((fx + 300, 406), p.sex, font=font(24), fill=blue)
    d.text((fx, 448), "16 HGT", font=font(18), fill=blue)
    d.text((fx + 72, 444), p.height, font=font(24), fill=blue)
    d.text((fx + 180, 448), "17 WGT", font=font(18), fill=blue)
    d.text((fx + 255, 444), f"{p.weight} lb", font=font(24), fill=blue)
    d.text((fx + 380, 448), "18 EYES", font=font(18), fill=blue)
    d.text((fx + 462, 444), p.eyes, font=font(24), fill=blue)
    d.text((fx, 492), "5 DD", font=font(18), fill=blue)
    d.text((fx + 55, 488), p.dl_dd, font=font(22), fill=blue)
    # Signature
    sig = Image.new("L", (360, 70), 0)
    sd = ImageDraw.Draw(sig)
    sd.text((5, 5), f"{p.first} {p.last}", font=font(44), fill=255)
    sig = sig.transform(sig.size, Image.AFFINE, (1, 0.35, -12, 0, 1, 0), resample=Image.BICUBIC)
    card.paste((30, 30, 70), (40, 520), sig)
    # Specimen marks
    over = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(over)
    od.text((W / 2 + 60, H / 2 + 20), "SPECIMEN", font=font(130), fill=(200, 30, 30, 70), anchor="mm")
    over = over.rotate(18, resample=Image.BICUBIC, center=(W / 2, H / 2))
    card = Image.alpha_composite(card.convert("RGBA"), over).convert("RGB")
    d = ImageDraw.Draw(card)
    d.rectangle([0, H - 40, W, H], fill=rgb)
    d.text((W / 2, H - 20), "SAMPLE - NOT A GOVERNMENT DOCUMENT", font=font(22), fill="white", anchor="mm")
    # Rounded corners
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, W - 1, H - 1], radius=36, fill=255)
    return card, mask


def make_dl(p, img_path, txt_path, seed):
    rng = random.Random(seed)
    card, mask = render_card(p)
    BW, BH = 1400, 1000
    bg = Image.new("RGB", (BW, BH), (118, 104, 88))
    noise = Image.effect_noise((BW, BH), 22).convert("RGB")
    bg = Image.blend(bg, noise, 0.12).filter(ImageFilter.GaussianBlur(1.2))
    # Soft shadow
    shadow = Image.new("L", (BW, BH), 0)
    ox, oy = 194 + rng.randint(-20, 20), 181 + rng.randint(-20, 20)
    shadow.paste(mask, (ox + 14, oy + 18))
    shadow = shadow.filter(ImageFilter.GaussianBlur(16))
    bg.paste((40, 34, 28), (0, 0), shadow.point(lambda v: v * 0.55))
    bg.paste(card, (ox, oy), mask)
    angle = rng.uniform(-3.0, 3.0)
    photo = bg.rotate(angle, resample=Image.BICUBIC, fillcolor=(118, 104, 88))
    # Uneven lighting
    light = Image.radial_gradient("L").resize((BW, BH)).point(lambda v: 255 - v // 5)
    photo = Image.composite(photo, Image.new("RGB", (BW, BH), (60, 55, 50)), light)
    photo = photo.filter(ImageFilter.GaussianBlur(0.6))
    photo.save(img_path, "JPEG", quality=86, optimize=True)
    with open(txt_path, "w") as fh:
        fh.write("\n".join(dl_lines(p)) + "\n")


# --------------------------------------------------------------------------

def main(out):
    dirs = {k: os.path.join(out, k) for k in
            ["DrivingLicenses", "W2Sample", "Form1040Sample", "Utility",
             os.path.join("extracted_text", "DrivingLicenses")]}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    written = []
    for i, p in enumerate(PEOPLE):
        img = os.path.join(dirs["DrivingLicenses"], f"{p.initials}DL.jpeg")
        make_dl(p, img, os.path.join(dirs[os.path.join("extracted_text", "DrivingLicenses")],
                                     f"{p.initials}DL.txt"), seed=1000 + i)
        w2 = os.path.join(dirs["W2Sample"], f"{p.file_stem}W2.pdf")
        make_w2(p, w2)
        f1040 = os.path.join(dirs["Form1040Sample"], f"{p.file_stem}Form1040.pdf")
        make_1040(p, f1040)
        written += [img, w2, f1040]
        for u in p.utilities:
            path = os.path.join(dirs["Utility"], u.file_name)
            make_utility(p, u, path)
            written.append(path)
    bank = os.path.join(out, "Realistic_Bank_Statements.pdf")
    make_bank_statements(bank)
    written.append(bank)
    for path in sorted(written):
        print(path)
    print(f"{len(written)} documents")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="data", help="output folder (default: data)")
    main(ap.parse_args().out)
