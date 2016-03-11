# Final Output and Reflection
### David Abrahams and Nur Shlapobersky

## Assessment evidence and interpretation

The primary assessment evidence for our project is the [poster](./poster.pdf). However, a substancial amount of work was done in order to produce the visuals shown on the poster.

Due to the nature of our project and the data, we had to take a lot of steps before we were ready to actually create the visuals and do the analysis for our poster. Many cells were missing, the original .csv was not in a format where we could easily create useful dataframes, clustering was non-trivial, and creating the visuals themselves was quite difficult. Our [IPython notebook](./collab.ipynb) shows all the code used to generate these visuals. It is quite difficult to read as we wrote it all incrementally over the last two weeks.

For example, once we have rearranged our dataset into a dataframe where each row is a country and each column is a statistic, we can use the following helper function in order to perform clustering, and spectral embedding. This function returns the objects necessary for our other helper functions to create plots.

```
def cluster_highest_pop_countries(df, filter=False, features=None, n=None, m=3, country_data_thresh=0, column_data_thresh=.8):
    """This helper function clusters the highest n populated countries into m clusters. It returns the cluster, a
    cluster dictionary creating using cluster_label_dict(), X_red - a spectrical embedding of the data into two dimensions,
    and the list of countries."""

    # if we need to filter out rows and columns with too many NaNs
    if filter:
        if features is None:
            features = df.columns
        sample = filter_df(df, features, n=n, country_data_thresh=country_data_thresh,
                           column_data_thresh=column_data_thresh)
    else:
        sample = df

    countries = sample.index

    # convert to a matrix and standardize it
    X = sample.as_matrix()
    X = StandardScaler().fit_transform(X)
    n_samples, n_features = X.shape

    # spectrally embed the matrix down to one axis in order to later reorder the clusters so that lower numbers
    # correspond to poorer countries, and high numbers correspond to richer ones.
    X_red = manifold.SpectralEmbedding(n_components=1).fit_transform(X)
    position_dic = {}
    for i in range(len(countries)):
        position_dic[countries[i]] = X_red[i, 0]

    # specreally embed to 2-d in order to plot later
    X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)

    print "Clustering using %i columns for %i countries..." % (sample.shape[1], sample.shape[0])

    # perform agglomerative clustering
    clustering = AgglomerativeClustering(linkage='ward', n_clusters=m)
    clustering.fit(X)

    # convert the cluster into a dictionary mapping from country to its cluster number
    clust_dict = cluster_label_dict(countries, clustering)

    # reorder the clusters so that lower clusters are poorer countries
    reorder_dictionary(clust_dict, position_dic)

    # return everything we'll need later
    return sample, clustering, clust_dict, X_red, countries, position_dic
```

Note that many of these lines of code point to other helper functions, which perform more simple tasks.

Once we have performed clustering and obtained a dictionary mapping from country name to its cluster, we can use this helper function to plot a colored map of the world:

```
def plot_map(clust_dict, reverse=False, save_name=''):
    '''Use dict of labels and clusters of countries to create a world map color-coded by label'''

    #Create map object and load world countries Shapefiles
    m = Basemap(projection='mill')
    m.readshapefile('Borders/world', name='countries', drawbounds=True)

    # Extract the country names from Basemap. Each country appears once for each of its polygons
    country_names = [shape_dict['NAME'] for shape_dict in m.countries_info]
    # remap the names of the countries to match the names of the countries in the World Bank DataBank
    country_names = remap_countries(country_names)
    ax = plt.gca()
    labels = [label for country, label in clust_dict.iteritems()]

    # clust_dict is a dictionary mapping from country name to its cluster number
    for country, label in clust_dict.iteritems():
        # if the clusters are in the wrong orientation (Africa is Blue, Europe is Red), invert them
        if reverse:
            label = max(labels) - label + min(labels)
        found = False
        # convert the cluster number to a matplotlib color
        color = map_to_color(float(label), min(labels), max(labels))
        #Countries with non-contiguous landmasses are constructed as multiple polygons

        # create a list of indicies of where this country occurrs in basemap
        country_indices = []
        for i, c in enumerate(country_names):
            if c.lower() in country.lower() or country.lower() in c.lower():
                country_indices.append(i)
                found = True

        if not found:
            print country + " was not found!"

        # plot the country's polygons
        segs = [m.countries[index] for index in country_indices]
        polys = [Polygon(seg, facecolor=color, edgecolor=color) for seg in segs]
        [ax.add_patch(poly) for poly in polys]
    plt.gcf().set_size_inches(20,10)
    if save_name: plt.gcf().savefig(os.path.join(cur_dir, 'img', save_name + ".eps"))

    plt.show()
    return country_names
```

## Changing the world

This project has the potential to raise awareness about the direction that America has moved in the past three decades. It establishess our global "location" using concrete metrics, and it allows us to know both what makes America different from the rest of the world, and how it has been changing.

## Learning Goals

We totally met our learning goals, and our stretch goals. We learned about and implemented an unsupervised clustering algorithm, as well as generated compelling and appealing visuals that tell an interesting story.
