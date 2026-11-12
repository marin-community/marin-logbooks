// Copyright The Marin Authors
// SPDX-License-Identifier: Apache-2.0

// Set PLAYWRIGHT_MODULE and CHROMIUM_EXECUTABLE if using a bundled runtime.
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const { pathToFileURL } = require('url');
const path = require('path');
(async () => {
  const options = { headless: true };
  if (process.env.CHROMIUM_EXECUTABLE) options.executablePath = process.env.CHROMIUM_EXECUTABLE;
  const browser = await chromium.launch(options);
  try {
    const page = await browser.newPage();
    await page.goto(pathToFileURL(path.join(__dirname, 'reading-guide.html')).href);
    await page.evaluate(() => document.fonts.ready);
    await page.pdf({
      path: path.join(__dirname, 'reading-guide.pdf'),
      printBackground: true,
      preferCSSPageSize: true,
      displayHeaderFooter: true,
      headerTemplate: '<span></span>',
      footerTemplate: '<div style="font-family:Arial;font-size:9px;width:100%;text-align:center;color:#555">Gradient and curvature experiments · <span class="pageNumber"></span></div>',
    });
  } finally {
    await browser.close();
  }
})();
