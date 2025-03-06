const scdl = require('soundcloud-downloader').default

const URL = process.argv[2]; // Second command-line argument

async function download(URL) {
    try {
        // Get track information
        const trackInfo = await scdl.getInfo(URL);
        const artworkURL = trackInfo?.artwork_url.replace('large', 't500x500') || 'No artwork available';
        const title = trackInfo?.title || 'Unknown title';
        const username = trackInfo?.user?.username || 'Unknown artist';
        const duration = trackInfo?.full_duration || 0;

        const jsonResponse = {
            songURL: URL,
            artworkURL: artworkURL,
            title: title,
            username: username,
            duration: duration
        };

        console.log(JSON.stringify(jsonResponse));
    } catch (error) {
        console.error('Error fetching:', URL, error);
        console.log(JSON.stringify({ error: 'Failed to fetch track info' }));
        process.exit(1); // Exit with an error code
    }
}

// Run the download function
download(URL);