var fs = require('fs');
var https = require('https');
var http = require('http');
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
    try {
        if (url.slice(0, 5) === 'https') {
            var req = https.get(url, function(res) {
                res.pipe(file);
            });
        } else {
            var req = http.get(url, function(res) {
                res.pipe(file);
            });
        }
    } catch (e) {
        console.error(e);
    }
}

async function googleImages(allImages) {
    const browser = await puppeteer.launch({headless: false});
    const page = await browser.newPage();
    var url = 'https://www.google.com/search?q=' + HERO + '&tbm=isch';
    var pagesToLoad = 0;

    await page.goto(url);

    async function load(pageLoaded, HERO) {
        //console.log(pageLoaded)
         //for (var i = 0; i < pagesToLoad; i++) {
            await page.evaluate((pageLoaded, HERO) => {
                var imgElements = document.getElementsByTagName('img');
                var valid = [];
                console.log(imgElements.length);
                for (e of imgElements) {
                    if (e.dataset && ((e.dataset.iurl || e.dataset.src) && e.alt === 'Image result for ' + HERO)) {
                        valid.push(e);
                        if (pageLoaded === 0) {
                            e.click();
                        }
                    }
                }
                if (valid.length > 0) {
                    valid[valid.length - 1].id = "lastClass" + pageLoaded;
                    document.getElementById("lastClass" + pageLoaded).click();
                    document.getElementById("lastClass" + pageLoaded).scrollIntoView();
                }
            }, pageLoaded, HERO);

            await page.waitFor(3000);
            if (pageLoaded > 0) {
                await load(pageLoaded - 1);
            } else {
                return true;
            }
        //}
    }

    var images = await load(pagesToLoad, HERO).then(async() => {
        var find = await page.evaluate((HERO) => {
            var imgElements = document.getElementsByTagName('img');
            var imgCount = 0;
            var imgUrls = [];
            for (element in imgElements) {
                var e = imgElements[element];
                var imgUrl;
                if (e.alt === 'Image result for ' + HERO) {
                    imgUrl = e.parentElement.parentElement.getAttribute('href');
                }
    
                if (imgUrl) {
                    if (!imgUrls.includes(imgUrl)) {
                        imgUrl = decodeURIComponent(imgUrl.slice(15, imgUrl.search("&imgrefurl")));
                        imgUrls.push(imgUrl);
                        imgCount++;
                    }
                }
            }
            console.log("Retrieved " + imgCount + " images");
            console.log(imgUrls);
            return imgUrls;
        }, HERO);
        
        return find;
    });

    await browser.close();
    return images;
}

googleImages().then((images) => {
    for (i in images) {
        // Saves as HERO<image number>.<file extension>
        downloadImage(images[i], HERO + i, 'png', HERO_PATH);
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