var fs = require('fs');
var https = require('https');
var jsdom = require("jsdom");
var { JSDOM } = jsdom;
var $ = require('jquery')(require('jsdom-no-contextify').jsdom().parentWindow);
var puppeteer = require('puppeteer');

var HERO = 'Reinhardt'
var HERO_PATH = './Images/' + HERO + '/';

function getDOM(res) {
    $.get(URL, function(html) {
        var dom = new JSDOM(html.toString());
        res(dom);
    });
}


/**
 * Downloads an image from a given url
 * @param {String} url image url
 * @param {String} filename name
 * @param {String} ext picture extension to save as
 * @param {String} dir directory to save image
 */
function downloadImage(url, filename, ext, dir) {
    var file = fs.createWriteStream(dir + filename + "." + ext);
    var req = https.get(url, function(res) {
        res.pipe(file)
    });
}

async function googleImages(allImages) {
    const browser = await puppeteer.launch({headless: true});
    const page = await browser.newPage();
    var url = 'https://www.google.com/search?q=' + HERO + '&tbm=isch';
    var pagesToLoad = 3;

    await page.goto(url);

    async function load() {
        for (var i = 0; i < pagesToLoad; i++) {
            await page.evaluate((i) => {
                var imgElements = document.getElementsByTagName('img');
                var valid = [];
                for (e of imgElements) {
                    if (e.dataset) {
                        valid.push(e);
                    }
                }
                var last = document.getElementsByTagName('img')[valid.length - 1].id = "lastClass" + i;
                document.getElementById("lastClass" + i).scrollIntoView();
            }, i);
            if (i < pagesToLoad - 1) {
                await page.waitFor(3000);
            }
        }
    }

    await load();
    
    var images = await page.evaluate(() => {
        var imgElements = document.getElementsByTagName('img');
        var imgCount = 0;
        var imgUrls = [];
        for (element in imgElements) {
            var e = imgElements[element];
            var imgUrl;
            if (e.dataset) {
                if (e.dataset.iurl) {
                    imgUrl = e.dataset.iurl;
                } else if (e.dataset.src) {
                    imgUrl = e.dataset.src;
                } 
            }
            
            if (imgUrl) {
                if (!imgUrls.includes(imgUrl)) {
                    imgUrls.push(imgUrl);
                    imgCount++;
                }
            }
        }
        console.log("Retrieved " + imgCount + " images");
        console.log(imgUrls);
        return imgUrls;
    });
    
    await browser.close();
    return images;
}

googleImages().then((images) => {
    for (i in images) {
        // Saves as HERO<image number>.<file extension>
        downloadImage(images[i], HERO + i, '.png', HERO_PATH);
    }
});


function main(dom) {
    //var document = dom.window.document;
    //console.log($(document).find('img')[0]);
}

getDOM((dom) => {
    //main(dom);
});



/*
getDOM((dom) => {
    console.log(dom);
});
*/
//downloadImage('https://pbs.twimg.com/profile_images/1216813945408966663/vkVajfRz_400x400.jpg')