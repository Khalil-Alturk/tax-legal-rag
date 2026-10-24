"""Generate the two fictional sample regulations in data/ (English + Arabic).

Run once locally:  python scripts/make_sample_pdfs.py
The PDFs are committed, so Docker/CI never need to run this.

Arabic needs a font with Arabic glyphs AND correct ToUnicode mapping, otherwise text
extraction returns broken ligatures. Windows Arial works; on Linux pass e.g.
--font-dir /usr/share/fonts/truetype/noto --font-file NotoNaskhArabic-Regular.ttf
"""

import argparse
import logging
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)

CSS = """
@font-face {{ font-family: body; src: url({font}); }}
* {{ font-family: body; }}
body {{ font-size: 11.5px; line-height: 1.5; }}
h1 {{ font-size: 17px; }} h2 {{ font-size: 13.5px; margin-top: 10px; }}
.note {{ font-size: 9.5px; color: #666; }}
"""

DISCLAIMER_EN = "FICTIONAL DOCUMENT for software testing only. Not real law or tax advice."

EN_PAGES = [
    f"""<h1>Republic of Examplia: Value Added Tax Regulation No. 7 of 2025</h1>
<p class="note">{DISCLAIMER_EN}</p>
<h2>Article 1: Definitions</h2>
<p>"Taxable supply" means any supply of goods or services made in Examplia in the course of
business. "Taxable person" means any person who is registered or required to register under
Article 4. "Tax period" means the period for which a VAT return is filed.</p>
<h2>Article 2: Standard Rate</h2>
<p>Value Added Tax is charged at the standard rate of 15 percent on the value of every taxable
supply, unless Article 3 applies.</p>
<h2>Article 3: Zero-Rated and Exempt Supplies</h2>
<p>The following supplies are zero-rated (taxed at 0 percent, with input tax recoverable):
exports of goods outside Examplia, international transport of passengers, and supplies of
qualifying medicines.</p>
<p>The following supplies are exempt (no VAT charged and no input tax recovery): residential
rent, life insurance, and margin-based financial services.</p>""",
    """<h2>Article 4: Registration</h2>
<p>Registration is mandatory for any person whose taxable supplies in the previous 12 months,
or expected in the next 12 months, exceed EXD 375,000. Voluntary registration is permitted
where taxable supplies or expenses exceed EXD 187,500. A person must apply within 30 days of
exceeding the mandatory threshold.</p>
<h2>Article 5: Filing and Payment</h2>
<p>Taxable persons with annual taxable supplies above EXD 40 million file monthly returns. All
other taxable persons file quarterly. The return and payment are due no later than the last
day of the month following the end of the tax period.</p>
<h2>Article 6: Penalties</h2>
<p>Failure to register on time: a fine of EXD 10,000. Late filing of a return: a fine of between
5 percent and 25 percent of the tax due. Late payment: 5 percent of the unpaid tax for each
month or part of a month of delay.</p>
<h2>Article 7: Record Keeping</h2>
<p>Taxable persons must keep invoices, books and records for at least 6 years from the end of
the tax period to which they relate.</p>""",
]

AR_PAGES = [
    """<div dir="rtl">
<h1>جمهورية إكزامبليا: نظام ضريبة الدخل على الشركات رقم 12 لسنة 2025</h1>
<p class="note">وثيقة خيالية لأغراض اختبار البرمجيات فقط، وليست قانونا حقيقيا ولا استشارة ضريبية.</p>
<h2>المادة الأولى: التعريفات</h2>
<p>الشركة المقيمة: كل شركة تأسست في إكزامبليا أو يقع فيها مركز إدارتها الفعلي.
الوعاء الضريبي: صافي الربح الخاضع للضريبة بعد خصم المصروفات الجائز خصمها.</p>
<h2>المادة الثانية: سعر الضريبة</h2>
<p>تفرض ضريبة الدخل على الشركات المقيمة بنسبة 20 بالمئة من الوعاء الضريبي.
وتخضع الشركات العاملة في استخراج النفط والغاز لسعر خاص قدره 50 بالمئة.</p>
<h2>المادة الثالثة: ضريبة الاستقطاع</h2>
<p>يجب على كل مقيم يدفع مبالغ إلى غير مقيم أن يستقطع الضريبة بالنسب الآتية:
5 بالمئة على أرباح الأسهم، و15 بالمئة على الإتاوات، و20 بالمئة على أتعاب الإدارة.
ويورد المبلغ المستقطع إلى الهيئة خلال 10 أيام من بداية الشهر التالي للدفع.</p>
</div>""",
    """<div dir="rtl">
<h2>المادة الرابعة: تقديم الإقرار</h2>
<p>يقدم الإقرار الضريبي السنوي وتسدد الضريبة المستحقة خلال 120 يوما من نهاية السنة المالية.</p>
<h2>المادة الخامسة: الخسائر</h2>
<p>يجوز ترحيل الخسائر الضريبية إلى السنوات التالية دون حد زمني، على ألا يتجاوز المبلغ المخصوم
في أي سنة 25 بالمئة من الربح الخاضع للضريبة لتلك السنة.</p>
<h2>المادة السادسة: الغرامات</h2>
<p>يعاقب على التأخر في تقديم الإقرار بغرامة قدرها 1 بالمئة من الضريبة المستحقة عن كل 30 يوما
من التأخير. ويعاقب على التهرب الضريبي بغرامة تعادل 300 بالمئة من الضريبة المتهرب منها.</p>
<h2>المادة السابعة: حفظ السجلات</h2>
<p>تلتزم الشركات بحفظ الدفاتر والسجلات والمستندات المؤيدة لمدة 10 سنوات.</p>
</div>""",
]


def build_pdf(path: Path, pages: list[str], font_dir: Path, font_file: str) -> None:
    """Render HTML pages into a PDF with an embedded font."""
    archive = pymupdf.Archive(str(font_dir))
    css = CSS.format(font=font_file)
    doc = pymupdf.open()
    for html in pages:
        page = doc.new_page(width=595, height=842)  # A4
        spare, _ = page.insert_htmlbox(
            page.rect + (50, 50, -50, -50), html, css=css, archive=archive
        )
        if spare < 0:
            raise ValueError(f"Content does not fit on one page in {path.name}")
    doc.set_metadata({"title": path.stem, "author": "Fictional sample"})
    doc.save(path, garbage=4, deflate=True)
    doc.close()
    logger.info("Wrote %s (%d pages)", path, len(pages))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--font-dir", type=Path, default=Path("C:/Windows/Fonts"))
    parser.add_argument("--font-file", default="arial.ttf")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    build_pdf(args.out / "vat_regulation_en.pdf", EN_PAGES, args.font_dir, args.font_file)
    build_pdf(args.out / "income_tax_regulation_ar.pdf", AR_PAGES, args.font_dir, args.font_file)


if __name__ == "__main__":
    main()
