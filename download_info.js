const scdl = require('soundcloud-downloader').default
const fs = require('fs')

const URL = process.argv[2]; // Second command-line argument

async function download(URL) {
    try {
        // Get track information
        const trackInfo = await scdl.getInfo(URL);
        const artworkURL = trackInfo?.artwork_url.replace('large', 't500x500') || 'No artwork available';
        const title = trackInfo?.title || 'No title available';
        const username = trackInfo?.user?.username || 'Unknown artist';

        const jsonResponse = {
            songURL: URL,
            artworkURL: artworkURL,
            title: title,
            username: username
        };

        // Print JSON to stdout
        
        console.log(JSON.stringify(jsonResponse));
        // process.exit(0); // Exit with an error code
    } catch (error) {
        console.error('Error fetching:', URL, error);
        console.log(JSON.stringify({ error: 'Failed to fetch track info' }));
        process.exit(1); // Exit with an error code
    }
}

// Run the download function
download(URL);