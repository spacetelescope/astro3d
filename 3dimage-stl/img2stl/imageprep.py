import numpy as np
from scipy import ndimage
from matplotlib.path import Path
import matplotlib.pyplot as plt
try:
	from astropy.io import fits
except ImportError:
	import pyfits as fits
import photutils

import meshcreator
import imageutils as iutils
import texture as _texture

def make_model(image, spiralarms, disk, clusters, height=150., stars=None):
		print image.shape
		print "Removing stars"
		if stars:
			image = remove_stars(image, stars)

		# Filter image
		print "Filtering image"
		image = ndimage.filters.median_filter(image, max(image.shape) / 100)
		image = np.ma.masked_equal(image, 0.0)
		image = iutils.normalize(image, True)

		print "Scaling top"
		image = scale_top(image, disk, spiralarms) # make sure spiralarms is one mask
		image = iutils.normalize(image, True)

		print image.shape
		print "Replacing cusp"
		image = replace_cusp(image)
		print "Emphasizing regions"
		image = emphasize_regions(image, (spiralarms, disk))
		print "Cropping image"
		image, (spiralarms, disk), clusters = crop_image_and_masks(image, (spiralarms, disk), clusters)
		print image.shape
		# Filter image again
		print "Filtering image"
		image = ndimage.filters.median_filter(image, 10) # Adjustable for image size?
		image = ndimage.filters.gaussian_filter(image, 3)
		image = np.ma.masked_equal(image, 0)
		image = iutils.normalize(image, True, height)

		print "Displaying clusters"
		clusters = reduce(add_clusters, # Fix add_clusters so taller clusters get added over shorter
				[make_star_cluster(image, cluster, clusters['flux'][0]) for cluster in clusters])
		clustermask = clusters != 0
		image[clustermask] = clusters[clustermask]

		print "Adding texture"
		texture = galaxy_texture(image, lmask=disk, dmask=spiralarms) # or lmask=disk, dmask=spiralarms
		texture[clustermask] = 0
		image = image + texture

		if isinstance(image, np.ma.core.MaskedArray):
			image = image.data

		print "Making base"
		image = image + make_base(image, 60, 10) # Should vary based on options
		return image

def prepFits(filename=None, array=None, height=150.0, spiralarms=None, disk=None, stars=None, clusters=None, 
	rotation=0.0, filter_radius=2, replace_stars=True, texture=True, num=15, remove=0):
	"""
	Prepares a fits file to be printed with a 3D printer.
		filename - a string specifying the path and name of the target fits file
		height - the maximum height above the base
		regions - squish values below the average height of input regions
		stars - a list of very bright objects (usually stars) that need to be removed in
				order for proper scaling
		rotation - # of degrees to rotate the image, usually to undo a previously rotated image
		filter_radius - the amount of smoothing to apply to the image. Keep between 2 and 5.
		replace_stars - replaces high values with an artificial star that is better for texture
		texture - automatically applies a certain texture to galaxies.
	"""
	# TODO: ratio is not a good indicator
	ratio = None
	img = array
	if not img:
		if not filename:
			raise ValueError("Must provide either filename or array")
		# Get file
		print "Getting file"
		img = fits.getdata(filename)
		img = np.flipud(img)
	h, w = img.shape

	if stars:
		print "Removing stars"
		if isinstance(stars, dict): stars = [stars]
		starmasks = [region_mask(img, star, True) for star in stars]
		for mask in starmasks:
			ypoints, xpoints = np.where(mask)
			dist = max(ypoints.ptp(), xpoints.ptp())
			newmasks = [img[ypoints+dist, xpoints], img[ypoints-dist, xpoints], 
						img[ypoints, xpoints+dist], img[ypoints, xpoints-dist]]
			medians = [newmask.mean() for newmask in newmasks]
			index = np.argmax(medians)
			img[mask] = newmasks[index]

	spiralarms = [region_mask(img, arm, True) for arm in spiralarms]
	disk = region_mask(img, disk, True)
	masks = spiralarms + [disk]
	if rotation:
		print "Rotating Image"
		if masks:
			masks = [ndimage.interpolation.rotate(mask.astype(int), rotation).astype(bool) for mask in masks]
			spiralarms = masks[:-1]
			disk = masks[-1]
		img = ndimage.interpolation.rotate(img, rotation)
		print "Cropping image"
		if masks:
			img, masks = iutils.crop_image(img, 1.0, masks)
			spiralarms = masks[:-1]
			disk = masks[-1]
		else:
			img = iutils.crop_image(img, 1.0)
	peaks = find_peaks(img, remove, num)
	# Filter values (often gets rid of stars), normalize
	print "Filtering image"
	img = ndimage.filters.median_filter(img, max(h, w) / 100)
	img = np.ma.masked_equal(img, 0.0)
	img = iutils.normalize(img, True)
	# Rescale very high values (cusp of galaxy, etc.)
	print "Scaling top"
	img = scale_top(img, disk, combine_masks(spiralarms))
	#img = scale_top_old(img)
	img = iutils.normalize(img, True)
	if replace_stars:
		print "Replacing stars"
		scale = 3
		jump = 1
		radius = None
		while True:
			top = img.mean() + scale * img.std()
			to_replace = np.where(img > top)
			ymin, xmin, ymax, xmax = to_replace[0].min(), to_replace[1].min(), to_replace[0].max(), to_replace[1].max()
			radius = max(xmax-xmin, ymax-ymin) / 2.
			if ratio == None: ratio = h / radius
			print radius
			if radius < 20:
				if jump > 0: jump *= -0.5
				scale += jump
			elif radius > 30:
				if jump < 0: jump *= -0.5
				scale += jump
			else:
				ratio = (h / float(radius)) / float(ratio)
				break
		star = make_star(radius, 40)
		img[ymin:ymin+2*radius+1, xmin:xmin+2*radius+1][star != -1] = top + star[star != -1]
	# Squish lower bound
	if spiralarms or disk:
		print "Squishing lower bound"
		img = emphasize_regions(img, masks)
	# Get rid of 'padding'
	print "Cropping image"
	if masks and clusters:
		img, masks, peaks = iutils.crop_image(img, 1.0, masks, peaks)
		spiralarms = masks[:-1]
		disk = masks[-1]
	elif masks:
		img, masks = iutils.crop_image(img, 1.0, masks)
		spiralarms = masks[:-1]
		disk = masks[-1]
	elif clusters:
		img, peaks = iutils.crop_image(img, 1.0, points=peaks)
	else:
		img = iutils.crop_image(img, 1.0)

	# Filter, smooth, normalize again
	print "Filtering image"
	img = ndimage.filters.median_filter(img, 10) # Needs to be adjustable for image size
	img = ndimage.filters.gaussian_filter(img, filter_radius)
	img = np.ma.masked_equal(img, 0)
	img = iutils.normalize(img, True, height)

	clustermask = None
	if clusters:
		print "Adding clusters"
		clusters = reduce(add_clusters, [make_star_cluster(img, peak, peaks['flux'][0]) for peak in peaks])
		clustermask = clusters != 0
		img[clustermask] = clusters[clustermask]
	if texture:
		print "Adding texture"
		#texture = galaxy_texture(img, lmask=disk, dmask=combine_masks(spiralarms))
		texture = galaxy_texture(img, 1.1)
		#texture = a_texture(img, masks)
		if clusters is not None: texture[clustermask] = 0
		img = img + texture
	if isinstance(img, np.ma.core.MaskedArray):
		img = img.data
	return img

def crop_image_and_masks(image, masks=None, peaks=None):
	if masks and peaks:
		image, masks, peaks = iutils.crop_image(image, 1.0, masks, peaks)
		spiralarms = masks[:-1]
		disk = masks[-1]
		return image, masks, peaks
	elif masks:
		image, masks = iutils.crop_image(image, 1.0, masks)
		spiralarms = masks[:-1]
		disk = masks[-1]
		return image, masks
	elif peaks:
		image, peaks = iutils.crop_image(image, 1.0, points=peaks)
		return image, peaks
	else:
		image = iutils.crop_image(image, 1.0)
		return image

def replace_cusp_old(image):
	scale = 3
	jump = 1
	radius = None
	ratio = None
	while True:
		top = image.mean() + scale * image.std()
		to_replace = np.where(image > top)
		ymin, ymax = to_replace[0].min(), to_replace[0].max()
		xmin, xmax = to_replace[1].min(), to_replace[1].max()
		radius = max(xmax - xmin, ymax - ymin) / 2.
		print 'radius = ', radius
		if ratio is None:
			ratio = image.shape[0] / radius
		if radius < 20:
			if jump > 0: jump *= -0.5
			scale += jump
		elif radius > 30:
			if jump < 0: jump *= -0.5
			scale += jump
		else:
			ratio = (image.shape[0] / float(radius)) / float(ratio)
			break
	star = make_star(radius, 40)
	image[ymin:ymin + 2*radius + 1, xmin:xmin + 2*radius + 1][star != -1] = top + star[star != -1]
	return image, ratio

def replace_cusp(image):
	y, x = np.where(image == image.max())
	if not np.isscalar(y):
		med = len(y) / 2
		y, x = y[med], x[med]
	print y, x
	radius = 20
	ymin, ymax = y - radius, y + radius
	xmin, xmax = x - radius, x + radius
	top = np.percentile(image[ymin:ymax, xmin:xmax], 10)
	star = make_star(radius, 40)
	image[ymin:ymin + 2*radius + 1, xmin:xmin + 2*radius + 1][star != -1] = top + star[star != -1]
	return image

def remove_stars(image, stars):
	if isinstance(stars, dict): stars = [stars]
	starmasks = [region_mask(image, star, True) for star in stars]
	for mask in starmasks:
		ypoints, xpoints = np.where(mask)
		dist = max(ypoints.ptp(), xpoints.ptp())
		newmasks = [image[ypoints+dist, xpoints], image[ypoints-dist, xpoints], 
					image[ypoints, xpoints+dist], image[ypoints, xpoints-dist]]
		medians = [newmask.mean() for newmask in newmasks]
		index = np.argmax(medians)
		image[mask] = newmasks[index]
	return image

def emphasize_regions(image, masks):
	"""Actually lowers areas outside spiral arm of galaxy"""
	_min = min(image[mask].mean() for mask in masks)
	_min -= image.std() / 2.
	image[image < _min] = image[image < _min] * (image[image < _min] / _min)
	# Repeat a second time
	_min = min(image[mask].mean() for mask in masks)
	_min -= image.std() / 2.
	image[image < _min] = image[image < _min] * (image[image < _min] / _min)
	# Remove low bound
	boolarray = image < 20 # & ~combine_masks(masks)
	print len(image[boolarray])
	image[boolarray] = 0
	return image

def make_star(radius, height):
	"""very similar to texture.make_star, creates a crator-like depression that can be used to represent a star"""
	x, y = np.meshgrid(np.arange(radius*2+1), np.arange(radius*2+1))
	r = np.sqrt((x-radius)**2 + (y-radius)**2)
	star = height/radius**2 * r**2
	star[r > radius] = -1
	return star

def scale_top(image, lmask, dmask):
	"""Linear scale of very high values of image"""
	mask = lmask & ~dmask
	top = np.percentile(image[mask], 30)
	topmask = image > top
	image[topmask] = top + (image[topmask] - top) * 10. / image.max()
	return image

def scale_top_old(image):
	top = image.mean() + image.std()
	image[image > top] = top + (image[image > top] - top) * 10. / image.max()
	return image

def galaxy_texture(galaxy, scale=None, lmask=None, dmask=None):
	"""
	Applies texture to the spiral arms and disk of galaxy.
		If lmask and dmask arguments are given, the textured areas are the masked regions,
		where lmask represents the disk and dmask is the spiral arms

		If scale is given, the texture masks are automatically generated. This generation works
		well with NGC3344 (the first test galaxy) but seems to do poorly with NGC1566 (the second).

	"""
	if (lmask is not None and dmask is None) or (dmask is not None and lmask is None):
		raise ValueError("Must provide two masks")
	if not scale or lmask is not None:
		scale = 1
	dotgrid = _texture.hex_grid(galaxy.shape, 7)
	dots = _texture.dots('linear', galaxy.shape, 5, 3.2, dotgrid)
	lines = _texture.lines('linear', galaxy.shape, 10, 20, 1.2, 0)
	maxfilt = ndimage.filters.maximum_filter(galaxy, 25)

	dotmask = maxfilt / 1.1 - galaxy
	dotmask[dotmask > 0] = 0
	dotmask[dotmask < 0] = 1
	dots = dots*dotmask
	dots[galaxy < 1] = 0
	if lmask is not None:
		dots[lmask] = 0
	else:
		dots[galaxy > galaxy.max() - scale * galaxy.std()] = 0
		#dots[galaxy > galaxy.max() - 1.1 * galaxy.std()] = 0

	linemask = maxfilt + 5
	linemask[linemask < galaxy.max()] = 1
	linemask[linemask > galaxy.max()] = 0
	lines = lines*linemask
	if lmask is not None:
		lines[~lmask] = 0
	else:
		lines[galaxy < galaxy.max() - scale * galaxy.std()] = 0
	filt = ndimage.filters.maximum_filter(lines, 10)
	dots[filt != 0] = 0
	where = np.where(lines)
	print where[0].ptp(), where[1].ptp()
	#lines[galaxy < galaxy.max() - 1.1 * galaxy.std()] = 0

	return dots + lines

def make_star_cluster(image, peak, max_intensity):
	x, y, intensity = peak['xcen'], peak['ycen'], peak['flux']
	radius = 15 + 5 * intensity / float(max_intensity)
	star = make_star(radius, 5)
	centers = [(y + 0.5 * radius * np.sqrt(3), x), (y - 0.5 * radius * np.sqrt(3), x + radius), (y - 0.5 * radius * np.sqrt(3), x - radius)]
	array = np.zeros(image.shape)
	try:
		_max = np.percentile(image[y - 2*radius:y + 2*radius, 
									x - 2*radius:x + 2*radius], 75)
		for (y, x) in centers:
			r = star.shape[0]
			y -= r / 2
			x -= r / 2
			array[y:y+r, x:x+r][star != -1] = _max + star[star != -1]
	except IndexError, ValueError:
		return array
	filt = ndimage.filters.maximum_filter(array, 10)
	mask = (filt > 0) & (image > filt) & (array == 0)
	array[mask] = filt[mask]
	return array

def region_mask(image, region, interpolate):
	"""Uses matplotlib.path.Path to generate a numpy boolean array, which can then be used as a mask for a region"""
	y, x = np.indices(image.shape)
	y, x = y.flatten(), x.flatten()
	points = np.vstack((x, y)).T
	polygon = Path([(p.x(), p.y()) for p in region.points()])
	mask = polygon.contains_points(points).reshape(image.shape)
	if interpolate:
		if interpolate == True:
			interpolate = (np.percentile(image[mask], 50), np.percentile(image[mask], 75))
		elif np.isscalar(interpolate):
			interpolate = (np.percentile(image[mask], 0), np.percentile(image[mask], interpolate))
		else:
			interpolate = (np.percentile(image[mask], interpolate[0]), np.percentile(image[mask], interpolate[1]))

		nmin, nmax = interpolate
		filtered = np.zeros(mask.shape)
		filtered[mask] = 1
		radius = min(axis.ptp() for axis in np.where(mask))
		filtered = ndimage.filters.maximum_filter(filtered, min(radius, image.shape[0] / 33))
		filtered = image * filtered
		mask = mask | ((filtered > nmin) & (filtered < nmax))
		maxfilt = ndimage.filters.maximum_filter(mask.astype(int), 3)
		mask = maxfilt != 0

	return mask

def combine_masks(masks):
	return reduce(lambda m1, m2: m1 | m2, masks)

def add_clusters(cluster1, cluster2):
	if cluster1[cluster2 != 0].min() < cluster2[cluster2 != 0].min():
		cluster1[cluster2 != 0] = cluster2[cluster2 != 0]
	else:
		cluster1[cluster1 == 0] = cluster2[cluster1 == 0]
	return cluster1

def find_peaks(image, remove=0, num=None):
	"""Identifies the brightest point sources in an image."""
	threshold = 8
	while threshold >=4:
		segm_img = photutils.detect_sources(image, snr_threshold=threshold, npixels=10, mask_val=0.0)
		isophot = photutils.segment_photometry(image, segm_img)
		if len(isophot['xcen']) >= 35:
			break
		else:
			threshold -= 1
	isophot.sort('flux')
	isophot.reverse()
	if remove:
		isophot.remove_rows([i for i in range(remove)])
	"""else:
					isophot.remove_rows([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13])
					isophot.remove_rows([1, 2, 5, 7, 13])"""
	return isophot[:num] if num else isophot

def make_base(image, dist, height=10):
	"""Used to create a stronger base for printing. Prevents model from shaking back and forth due to printer vibration."""
	max_filt = ndimage.filters.maximum_filter(image, dist)
	max_filt[max_filt < 1] = -5
	max_filt[max_filt > 1] = 0
	max_filt[max_filt < 0] = height
	return max_filt

def prepareImg(filename, height=30, filter_radius=None, crop=False, invert=False, compress=True):
	"""An old method, used for testing img2stl.to_mesh on random images"""
	img = None
	if filename[-5:] == '.fits':
		f = fits.open(filename)
		for hdu in f:
			if isinstance(hdu.data, np.ndarray):
				img = hdu.data
				break
		f.close()
	else:
		img = iutils.img2array(filename)
	if crop != False:
		img = iutils.crop_image(img, crop) if np.isscalar(crop) else iutils.crop_image(img, 1.0)
		if np.isscalar(crop):
			img = remove_background(img, crop)
		else:
			img = remove_background(img, 1.0)

	if compress and img.shape[0] > 500:
		img = iutils.compressImage(img, 500)
	if filter_radius:
		img = ndimage.filters.gaussian_filter(img, filter_radius)
	img = img - img.min()
	if invert:
		img = img.max() - img
	img = iutils.normalize(img, True, height)
	return np.fliplr(img)
















